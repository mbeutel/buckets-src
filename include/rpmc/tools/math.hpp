
// Common mathematical constants and functions.


#ifndef INCLUDED_RPMC_TOOLS_MATH_HPP_
#define INCLUDED_RPMC_TOOLS_MATH_HPP_


#include <span>
#include <cmath>
#include <limits>
#include <utility>    // for pair<>, move()
#include <numeric>    // for accumulate()
#include <concepts>   // for arithmetic<>, floating_point<>
#include <algorithm>  // for transform_reduce()

#include <gsl-lite/gsl-lite.hpp>  // for gsl_Expects(), gsl_ExpectsDebug()

#include <makeshift/tuple.hpp>  // for template_for()

#include <intervals/sign.hpp>

//#include <rpmc/tools/utility.hpp>  // for KahanAccumulator<>


namespace rpmc {

namespace gsl = ::gsl_lite;


    // Math constants
constexpr double pi = 3.14159265358979323846;


template <typename ForwardIt, std::floating_point FloatT>
struct DiscreteInverseTransformResult
{
    ForwardIt pos;
    FloatT residual;
};

template <typename ForwardIt, typename ForwardEndIt, std::floating_point FloatT>
constexpr DiscreteInverseTransformResult<ForwardIt, FloatT>
discreteInverseTransform(
    ForwardIt first, ForwardEndIt last,
    FloatT val)  // value in [0, N) where `N = std::accumulate(first, last, FloatT(0))`
{
    FloatT x = 0.;
    //KahanAccumulator<FloatT> x = 0.;
    while (first != last)
    {
        auto delta = *first;
        auto xold = x;
        x += delta;
        if (x > val)
        {
            return {
                .pos = first,
                .residual = (val - xold)/(x - xold)
            };
        }
        ++first;
    }
    return {
        .pos = first,
        .residual = std::numeric_limits<FloatT>::quiet_NaN()
    };
}

template <typename ForwardIt, typename ForwardEndIt, std::floating_point FloatT, typename ProjT>
constexpr DiscreteInverseTransformResult<ForwardIt, FloatT>
discreteInverseTransform(
    ForwardIt first, ForwardEndIt last,
    FloatT val,  // value in [0, N) where `N = std::transform_reduce(first, last, FloatT(0), std::plus<>{ }, proj)`
    ProjT proj)
{
    FloatT x = 0.;
    //KahanAccumulator<FloatT> x = 0.;
    while (first != last)
    {
        auto delta = proj(*first);
        auto xold = x;
        x += delta;
        if (x > val)
        {
            return {
                .pos = first,
                .residual = (val - xold)/(x - xold)
            };
        }
        ++first;
    }
    return {
        .pos = first,
        .residual = std::numeric_limits<FloatT>::quiet_NaN()
    };
}


} // namespace rpmc


#endif // INCLUDED_RPMC_TOOLS_MATH_HPP_
