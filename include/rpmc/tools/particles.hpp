
#ifndef INCLUDED_RPMC_TOOLS_PARTICLES_HPP_
#define INCLUDED_RPMC_TOOLS_PARTICLES_HPP_


#include <span>
#include <tuple>
#include <memory>   // for unique_ptr<>
#include <utility>  // for swap()

#include <gsl-lite/gsl-lite.hpp>  // for index, not_null<>, make_unique<>()

#include <makeshift/tuple.hpp>                   // for apply(), tuple_cat()
#include <makeshift/experimental/algorithm.hpp>  // for apply_permutation() (TODO: hoist to non-experimental)

#if gsl_CPP20_OR_GREATER
# include <concepts>
#endif // gsl_CPP20_OR_GREATER


namespace rpmc {

namespace gsl = ::gsl_lite;


#if gsl_CPP20_OR_GREATER
template <typename ParticleDataT>
concept ParticleData = requires(ParticleDataT const& data, gsl::index iSrc, gsl::index iDst, std::span<gsl::index const> permutation)
{
    data.clone(iSrc, iDst);
    data.swap(iSrc, iDst);
    data.applyPermutation(permutation);
    data.remove(iDst);
};
#endif // gsl_CPP20_OR_GREATER


class PParticleData
{
private:
    struct IConcept
    {
        gsl::dim num;

        IConcept(gsl::dim _num)
            : num(_num)
        {
        }

        virtual ~IConcept() { }
        virtual void clone(gsl::index iSrc, gsl::index iDst) const = 0;
        virtual void swap(gsl::index j, gsl::index k) const = 0;
        virtual void applyPermutation(std::span<gsl::index const> permutation) const = 0;
        virtual void remove(gsl::index i) const = 0;
    };
    template <typename T>
    struct Model final : IConcept
    {
        T impl_;

        Model(T&& _impl) : IConcept(_impl.num()), impl_(std::move(_impl)) { }

        void
        clone(gsl::index iSrc, gsl::index iDst) const override
        {
            impl_.clone(iSrc, iDst);
        }
        void
        swap(gsl::index j, gsl::index k) const override
        {
            impl_.swap(j, k);
        }
        void
        applyPermutation(std::span<gsl::index const> permutation) const override
        {
            impl_.applyPermutation(permutation);
        }
        void
        remove(gsl::index i) const override
        {
            impl_.remove(i);
        }
    };

    gsl::not_null<std::shared_ptr<IConcept>> impl_;

public:
    template <ParticleData T>
    PParticleData(T data)
        : impl_(gsl::not_null(gsl::make_shared<Model<T>>(std::move(data))))
    {
    }
    gsl::dim
    num() const
    {
        return impl_->num;
    }
    void
    clone(gsl::index iSrc, gsl::index iDst) const
    {
        impl_->clone(iSrc, iDst);
    }
    void
    swap(gsl::index j, gsl::index k) const
    {
        impl_->swap(j, k);
    }
    void
    applyPermutation(std::span<gsl::index const> permutation) const
    {
        impl_->applyPermutation(permutation);
    }
    void
    remove(gsl::index i) const
    {
        impl_->remove(i);
    }
};


template <typename IdSpansT, typename PropertySpansT>
class SpanParticleData;
template <typename... IdSpansT, typename... PropertySpansT>
class SpanParticleData<std::tuple<IdSpansT...>, std::tuple<PropertySpansT...>>
{
private:
    gsl::dim num_;
    std::tuple<IdSpansT...> idSpans_;
    std::tuple<PropertySpansT...> propertySpans_;

public:
    SpanParticleData(std::tuple<IdSpansT...> _idSpans, std::tuple<PropertySpansT...> _propertySpans)
        : idSpans_(std::move(_idSpans)), propertySpans_(std::move(_propertySpans))
    {
        static_assert(sizeof...(IdSpansT) > 0 || sizeof...(PropertySpansT) > 0, "at least one data column must exist");
        if constexpr (sizeof...(IdSpansT) > 0)
        {
            num_ = gsl::ssize(std::get<0>(idSpans_));
        }
        else if constexpr (sizeof...(PropertySpansT) > 0)
        {
            num_ = gsl::ssize(std::get<0>(propertySpans_));
        }
        bool sameExtent = makeshift::apply(
            [this]
            (auto const&... spans)
            {
                return ((gsl::ssize(spans) == num_) && ...);
            },
            makeshift::tuple_cat(idSpans_, propertySpans_));
        gsl_Expects(sameExtent);
    }
    gsl::dim
    num() const
    {
        return num_;
    }
    void
    clone(gsl::index iSrc, gsl::index iDst) const
    {
        // Do not alter `idSpans_` when cloning particle properties.
        std::apply(
            [iSrc, iDst]
            (auto&... spans)
            {
                ((spans[iDst] = spans[iSrc]), ...);
            },
            propertySpans_);
    }
    void
    swap(gsl::index j, gsl::index k) const
    {
        std::apply(
            [j, k]
            (auto&... spans)
            {
                using std::swap;
                (swap(spans[j], spans[k]), ...);
            },
            idSpans_);
        std::apply(
            [j, k]
            (auto&... spans)
            {
                using std::swap;
                (swap(spans[j], spans[k]), ...);
            },
            propertySpans_);
    }
    void
    applyPermutation(std::span<gsl::index const> permutation) const
    {
            // `apply_permutation()` is destructive, i.e. it un-permutes the index array as it goes through the given range.
            // We therefore need to draw a fresh copy of the permutation span for each pass.
            // (I reckon this approach is more cache-friendly than zipping up all spans with `makeshift::range_zip()` and
            // permuting them all together.)
        auto lpermutation = std::vector<gsl::index>(permutation.size());

        std::apply(
            [permutation, &lpermutation]
            (auto&... spans)
            {
                ((std::copy(permutation.begin(), permutation.end(), lpermutation.begin()), makeshift::apply_permutation(spans.begin(), spans.end(), lpermutation.begin())), ...);
            },
            idSpans_);
        std::apply(
            [permutation, &lpermutation]
            (auto&... spans)
            {
                ((std::copy(permutation.begin(), permutation.end(), lpermutation.begin()), makeshift::apply_permutation(spans.begin(), spans.end(), lpermutation.begin())), ...);
            },
            propertySpans_);
    }
    void
    remove(gsl::index i) const
    {
        // Do not alter `idSpans_` when marking a particle inactive.
        std::apply(
            [i]
            (auto&... spans)
            {
                ((spans[i] = { }), ...);
            },
            propertySpans_);
    }
};
template <typename... IdSpansT, typename... PropertySpansT>
SpanParticleData(std::tuple<IdSpansT...>, std::tuple<PropertySpansT...>) -> SpanParticleData<std::tuple<IdSpansT...>, std::tuple<PropertySpansT...>>;


} // namespace rpmc


#endif // INCLUDED_RPMC_TOOLS_PARTICLES_HPP_
