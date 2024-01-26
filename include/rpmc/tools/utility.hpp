
// ??


#ifndef INCLUDED_RPMC_TOOLS_UTILITY_HPP_
#define INCLUDED_RPMC_TOOLS_UTILITY_HPP_


#include <span>
#include <limits>
#include <concepts>     // for integral<>
#include <algorithm>    // for min(), max()
#include <type_traits>  // for is_signed<>

#include <gsl-lite/gsl-lite.hpp>  // for gsl_Expects(), gsl_Assert()


namespace rpmc {

namespace gsl = gsl_lite;


    // Simpler definition of `span<>` as a single-argument template.
template <typename T> using Span = std::span<T>;


namespace detail {


    // Computes ⌊√v⌋.
    // This function is implemented with recursion, which is probably unsatisfactory for runtime performance, so we use it only
    // internally, and only for compile-time computations.
template <typename V>
constexpr V
sqrti(V v)
{
    gsl_Expects(v >= 0);

    V a = 0;
    if (v < 2) return v;
    a = sqrti<V>(v / 4) * 2; // a² ≤ v
    return v - a*a < 2*a + 1 // equivalent to `(a + 1)² > v` but without the possibility of overflow
        ? a
        : a + 1;
}


} // namespace detail


template <std::integral T>
constexpr T
square_checked_failfast(T v)
{
    constexpr T m = detail::sqrti(std::numeric_limits<T>::max());
    gsl_Assert(v <= m && (!std::is_signed_v<T> || v >= -m));
    return v*v;
}


template <bool Value, typename A, typename B>
[[nodiscard]] constexpr decltype(auto)
if_else_c(std::bool_constant<Value>, A&& a, B&& b)
{
    if constexpr(Value)
    {
        return std::forward<A>(a);
    }
    else
    {
        return std::forward<B>(b);
    }
}


template <std::floating_point T>
class KahanAccumulator
{
private:
    T value_;
    T carry_;

    explicit constexpr KahanAccumulator(T _value, T _carry)
        : value_(_value), carry_(_carry)
    {
    }

public:
    constexpr KahanAccumulator()
        : value_{ },
          carry_{ }
    {
    }
    constexpr KahanAccumulator(T _value)
        : value_(_value),
          carry_{ }
    {
    }

    constexpr KahanAccumulator&
    operator =(T x)
    {
        value_ = x;
        carry_ = 0;
        return *this;
    }

    constexpr KahanAccumulator&
    operator +=(T x)
    {
        // Borrowed from the Wikipedia page on Kahan summation, https://en.wikipedia.org/wiki/Kahan_summation_algorithm .
    
        T y = x - carry_;
        T newValue = value_ + y;
        carry_ = (newValue - value_) - y;
        value_ = newValue;
        return *this;
    }
    constexpr KahanAccumulator&
    operator -=(T x)
    {
        return (*this += -x);
    }

    constexpr KahanAccumulator&
    operator +=(KahanAccumulator const& x)
    {
        T y = x.value_ - carry_;
        T newValue = value_ + y;
        carry_ = (newValue - value_) - y + x.carry_;
        value_ = newValue;
        return *this;
    }
    constexpr KahanAccumulator&
    operator -=(KahanAccumulator const& x)
    {
        T y = -x.value_ - carry_;
        T newValue = value_ + y;
        carry_ = (newValue - value_) - y - x.carry_;
        value_ = newValue;
        return *this;
    }

    friend constexpr KahanAccumulator
    operator +(KahanAccumulator const& lhs, KahanAccumulator const& rhs)
    {
        KahanAccumulator result = lhs;
        result += rhs;
        return result;
    }
    friend constexpr KahanAccumulator
    operator -(KahanAccumulator const& lhs, KahanAccumulator const& rhs)
    {
        KahanAccumulator result = lhs;
        result -= rhs;
        return result;
    }

    constexpr T
    value() const
    {
        return value_;
    }
    explicit constexpr
    operator T() const
    {
        return value_;
    }

    friend constexpr bool
    operator ==(KahanAccumulator const& lhs, KahanAccumulator const& rhs)
    {
        return lhs.value_ == rhs.value_;
    }
    friend constexpr auto
    operator <=>(KahanAccumulator const& lhs, KahanAccumulator const& rhs)
    {
        return lhs.value_ <=> rhs.value_;
    }

    friend constexpr KahanAccumulator
    min(KahanAccumulator const& lhs, KahanAccumulator const& rhs)
    {
        return lhs.value_ < rhs.value_ ? lhs
            : lhs.value_ > rhs.value_ ? rhs
            : KahanAccumulator{ lhs.value_, std::max(lhs.carry_, rhs.carry_) };
    }
    friend constexpr KahanAccumulator
    max(KahanAccumulator const& lhs, KahanAccumulator const& rhs)
    {
        return lhs.value_ > rhs.value_ ? lhs
            : lhs.value_ < rhs.value_ ? rhs
            : KahanAccumulator{ lhs.value_, std::min(lhs.carry_, rhs.carry_) };
    }
};


} // namespace rpmc


#endif // INCLUDED_RPMC_TOOLS_UTILITY_HPP_
