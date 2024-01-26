
#ifndef INCLUDED_RPMC_DETAIL_CHECKED_HPP_
#define INCLUDED_RPMC_DETAIL_CHECKED_HPP_


#include <limits>
#include <stdexcept>  // for overflow_error

#include <gsl-lite/gsl-lite.hpp>  // for gsl_Expects()


namespace rpmc {

namespace gsl = gsl_lite;

namespace detail {


template <typename T>
[[nodiscard]] constexpr T
checkedAddSize(T a, T b)
{
    gsl_Expects(a >= 0 && b >= 0);
    if (a > std::numeric_limits<T>::max() - b) throw std::overflow_error("integer overflow");
    return a + b;
}
template <typename T>
[[nodiscard]] constexpr T
checkedMultiplySize(T a, T b)
{
    gsl_Expects(a >= 0 && b >= 0);
    if (b != 0 && a > std::numeric_limits<T>::max()/b) throw std::overflow_error("integer overflow");
    return a*b;
}


} // namespace detail

} // namespace rpmc


#endif // INCLUDED_RPMC_DETAIL_CHECKED_HPP_
