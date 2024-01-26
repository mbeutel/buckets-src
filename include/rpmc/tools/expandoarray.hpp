
#ifndef INCLUDED_RPMC_TOOLS_EXPANDOARRAY_HPP_
#define INCLUDED_RPMC_TOOLS_EXPANDOARRAY_HPP_


#include <array>
#include <limits>
#include <ranges>
#include <memory>       // for unique_ptr<>
#include <utility>      // for move()
#include <cstdint>      // for uint32_t
#include <algorithm>    // for max(), copy(), fill()
#include <type_traits>  // for integral_constant<>, is_signed<>

#include <gsl-lite/gsl-lite.hpp>  // for dim, stride, index

#include <makeshift/array.hpp>
//#include <makeshift/algorithm.hpp>  // for range_zip()
//#include <makeshift/ranges.hpp>

#include <rpmc/detail/checked.hpp>  // for checkedAddSize(), checkedMultiplySize()


namespace rpmc {

namespace gsl = ::gsl_lite;


template <typename T, std::size_t NumAxes, typename IndexT = gsl::index>
class ExpandoArray
{
    static_assert(NumAxes <= 32);
    static_assert(std::is_signed_v<IndexT>);

    using Index = IndexT;
    using Dim = IndexT;

private:
    static constexpr gsl::dim numAxes_ = NumAxes;

    struct AxisData
    {
        Index firstIndex;
        Index lastIndex;
        gsl::stride stride;
    };

    std::unique_ptr<T[]> data_;
    std::array<AxisData, NumAxes> axisData_;
    std::uint32_t fixedAxisFlags_;

    static Dim
    grow(Dim oldSize, Dim newSize)
    {
        if (oldSize > std::numeric_limits<Dim>::max() - oldSize/3 - 1)
        {
            return newSize;
        }
        return std::max(gsl::narrow_failfast<Dim>(oldSize + oldSize/3 + 1), newSize);
    }

    static void
    move(
        T* dst, T const* src,
        std::array<AxisData, NumAxes> const& /*dstAxisData*/, std::array<AxisData, NumAxes> const& /*srcAxisData*/,
        std::integral_constant<std::size_t, NumAxes>) noexcept
    {
        *dst = std::move(*src);
    }
    template <std::size_t I>
    static void
    move(
        T* dst, T const* src,
        std::array<AxisData, NumAxes> const& dstAxisData, std::array<AxisData, NumAxes> const& srcAxisData,
        std::integral_constant<std::size_t, I>) noexcept
    {
        Dim prefix = srcAxisData[I].firstIndex - dstAxisData[I].firstIndex;
        Dim size = srcAxisData[I].lastIndex - srcAxisData[I].firstIndex;
        if constexpr (I + 1 == NumAxes)
        {
            dst += prefix;
            std::move(src, src + size, dst);
        }
        else
        {
            gsl::stride dstStride = dstAxisData[I].stride;
            gsl::stride srcStride = srcAxisData[I].stride;
            dst += prefix*dstStride;
            for (Index k = 0; k != size; ++k)
            {
                move(dst, src, dstAxisData, srcAxisData, std::integral_constant<std::size_t, I + 1>{ });
                dst += dstStride;
                src += srcStride;
            }
        }
    }

    void
    expandBy(std::array<Index, NumAxes> const& delta)
    {
        auto newAxisData = axisData_;
        gsl::stride stride = 1;
        for (Index axis = numAxes_ - 1; axis >= 0; --axis)  // C order
        {
            if (delta[axis] >= 0)
            {
                newAxisData[axis].lastIndex += delta[axis];
            }
            else
            {
                newAxisData[axis].firstIndex += delta[axis];
            }
            newAxisData[axis].stride = stride;
            Dim size = newAxisData[axis].lastIndex - newAxisData[axis].firstIndex;
            stride = detail::checkedMultiplySize(stride, gsl::dim(size));
        }
        auto newData = std::make_unique<T[]>(stride);
        move(newData.get(), data_.get(), newAxisData, axisData_, std::integral_constant<std::size_t, 0>{ });
        data_ = std::move(newData);
        axisData_ = newAxisData;
    }

    void
    expandTo(std::array<Index, NumAxes> const& index)
    {
        auto delta = std::array<Index, NumAxes>{ };
        for (Index axis = 0; axis != numAxes_; ++axis)
        {
            auto& axisData = axisData_[axis];
            Dim size = axisData.lastIndex - axisData.firstIndex;
            if (size == 0)
            {
                    // The initial allocation can be placed anywhere.
                axisData.firstIndex = axisData.lastIndex = index[axis];
                delta[axis] = 1;
            }
            else if (index[axis] >= axisData.lastIndex)
            {
                Dim suffix = index[axis] + 1 - axisData.lastIndex;
                Dim requestedSize = detail::checkedAddSize(size, suffix);
                Dim newSize = grow(size, requestedSize);
                delta[axis] = newSize - size;  // > 0
            }
            else if (index[axis] < axisData.firstIndex)
            {
                Dim prefix = axisData.firstIndex - index[axis];
                Dim requestedSize = detail::checkedAddSize(size, prefix);
                Dim newSize = grow(size, requestedSize);
                delta[axis] = size - newSize;  // < 0
            }
            gsl_Assert(delta[axis] == 0 || (fixedAxisFlags_ & (std::uint32_t(1) << axis)) == 0);
        }
        expandBy(delta);
    }

    gsl::index
    computeFlatIndex(std::array<Index, NumAxes> const& index) const
    {
        gsl::index flatIndex = 0;
        for (Index axis = 0; axis < numAxes_; ++axis)
        {
            if (index[axis] < axisData_[axis].firstIndex || index[axis] >= axisData_[axis].lastIndex)
            {
                return -1;
            }
            flatIndex += (index[axis] - axisData_[axis].firstIndex) * axisData_[axis].stride;
        }
        return flatIndex;
    }
    gsl::dim
    computeFlatSize() const
    {
        if constexpr (NumAxes == 0)
        {
            return 1;
        }
        else
        {
            return (axisData_[0].lastIndex - axisData_[0].firstIndex) * axisData_[0].stride;
        }
    }

public:
    constexpr ExpandoArray()
        : data_{ }, axisData_{ }, fixedAxisFlags_(0)
    {
        if constexpr (NumAxes == 0)
        {
            data_ = std::make_unique<T[]>(1);
        }
    }
    explicit constexpr ExpandoArray(std::array<Dim, NumAxes> _fixedShape)
        : fixedAxisFlags_(0)
    {
        gsl::stride stride = 1;
        for (Index axis = numAxes_ - 1; axis >= 0; --axis)  // C order
        {
            Dim size = _fixedShape[axis] >= 0 ? _fixedShape[axis] : 0;
            axisData_[axis] = {
                .firstIndex = 0,
                .lastIndex = size,
                .stride = stride
            };
            if (_fixedShape[axis] >= 0)
            {
                fixedAxisFlags_ |= (std::uint32_t(1) << axis);
            }
            stride = detail::checkedMultiplySize(stride, gsl::dim(size));
        }
        if (stride != 0)  // all dimensions fixed
        {
            data_ = std::make_unique<T[]>(stride);
        }
    }

    T const&
    operator [](std::array<Index, NumAxes> const& index) const
    {
        gsl::index flatIndex = computeFlatIndex(index);
        gsl_Assert(flatIndex >= 0);
        return data_[flatIndex];
    }
    T&
    operator [](std::array<Index, NumAxes> const& index)
    {
        gsl::index flatIndex = computeFlatIndex(index);
        gsl_Assert(flatIndex >= 0);
        return data_[flatIndex];
    }

    T&
    obtain(std::array<Index, NumAxes> const& index)
    {
        gsl::index flatIndex = computeFlatIndex(index);
        if (flatIndex < 0)
        {
            expandTo(index);
            flatIndex = computeFlatIndex(index);
            gsl_Assert(flatIndex >= 0);
        }
        return data_[flatIndex];
    }
    T&
    assign(std::array<Index, NumAxes> const& index, T value)
    {
        return (obtain(index) = std::move(value));  // we're being lazy here and rely on default initialization
    }

    void
    remove(std::array<Index, NumAxes> const& index)
    {
        gsl::index flatIndex = computeFlatIndex(index);
        if (flatIndex >= 0)
        {
            data_[flatIndex] = { };
        }
        // TODO: would we attempt to actually shrink the array?
    }

    template <typename F>
    void
    visit(F&& func)
    {
        for (T* it = data_.get(), * last = it + computeFlatSize(); it != last; ++it)
        {
            func(*it);
        }
    }
    void
    fill(T const& value = { })
    {
        for (T* it = data_.get(), * last = it + computeFlatSize(); it != last; ++it)
        {
            *it = value;
        }
    }

    auto
    entries() const
    {
        auto first = data_.get();
        auto last = first + computeFlatSize();
        return std::ranges::subrange(first, last);
    }
    std::tuple<std::array<Index, NumAxes>, std::array<Index, NumAxes>>
    entryIndexRange() const
    {
            // not currently implemented for higher dimensions
            // (would be a lot of effort for something I don't need right now)
        static_assert(NumAxes == 1);

        Index firstIndex = axisData_[0].firstIndex;
        Index lastIndex = firstIndex + gsl::narrow_cast<Index>(computeFlatSize());
        return { { firstIndex }, { lastIndex } };
    }
    auto
    indexedEntries() const
    {
            // not currently implemented for higher dimensions
            // (would be a lot of effort for something I don't need right now)
        static_assert(NumAxes == 1);

        gsl::dim num = computeFlatSize();
        auto firstSize = data_.get();
        gsl::index firstIndex = axisData_[0].firstIndex;
        return std::views::iota(0, num)
            | std::views::transform(
                [firstSize, firstIndex]
                (gsl::index i)
                {
                    return std::tuple{ std::array{ gsl::narrow_cast<Index>(firstIndex + i) }, firstSize[i] };
                });
    }
    auto
    entriesInRange(std::array<Index, NumAxes> const& from, std::array<Index, NumAxes> const& to) const
    {
            // not currently implemented for higher dimensions
            // (would be a lot of effort for something I don't need right now)
        static_assert(NumAxes == 1);

        auto first = data_.get();
        gsl::dim maxNum = computeFlatSize();
        gsl::dim num = to[0] - from[0];
        gsl_Assert(num >= 0);
        gsl::diff delta = from[0] - axisData_[0].firstIndex;
        if (delta < 0)
        {
            num = std::min(maxNum, std::max(gsl::dim(0), num + delta));
        }
        else if (delta <= maxNum)
        {
            first += delta;
            num = std::min(num, maxNum - delta);
        }
        else
        {
            num = 0;
        }
        auto last = first + num;
        return std::ranges::subrange(first, last);
    }
    std::tuple<std::array<Index, NumAxes>, std::array<Index, NumAxes>>
    entryIndexRangeInRange(std::array<Index, NumAxes> const& from, std::array<Index, NumAxes> const& to) const
    {
            // not currently implemented for higher dimensions
            // (would be a lot of effort for something I don't need right now)
        static_assert(NumAxes == 1);

        gsl::dim maxNum = computeFlatSize();
        gsl::dim num = to[0] - from[0];
        gsl_Assert(num >= 0);
        gsl::diff delta = from[0] - axisData_[0].firstIndex;
        Index firstIndex = axisData_[0].firstIndex;
        if (delta < 0)
        {
            num = std::min(maxNum, std::max(gsl::dim(0), num + delta));
        }
        else if (delta <= maxNum)
        {
            firstIndex += gsl::narrow_cast<Index>(delta);
            num = std::min(num, maxNum - delta);
        }
        else
        {
            num = 0;
        }
        Index lastIndex = firstIndex + gsl::narrow_cast<Dim>(num);
        return { { firstIndex }, { lastIndex } };
    }
};


} // namespace rpmc


#endif // INCLUDED_RPMC_TOOLS_EXPANDOARRAY_HPP_
