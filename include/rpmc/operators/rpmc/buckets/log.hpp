
#ifndef INCLUDED_RPMC_OPERATORS_RPMC_BUCKETS_LOG_HPP_
#define INCLUDED_RPMC_OPERATORS_RPMC_BUCKETS_LOG_HPP_


#include <numbers>     // for ln10
#include <concepts>    // for floating_point<>

#include <gsl-lite/gsl-lite.hpp>  // for gsl_Expects(), gsl_ExpectsDebug()

#include <intervals/math.hpp>
#include <intervals/interval.hpp>
#include <intervals/type_traits.hpp>


namespace rpmc {

namespace gsl = gsl_lite;


template <typename T, std::floating_point U>
constexpr intervals::interval<T>
widenMultiplicatively(intervals::interval<T> const& x, U factor)
{
    gsl_ExpectsDebug(factor >= 1);

    return intervals::interval<T>{ x.lower()/factor, x.upper_unchecked()*factor };
}
template <typename T, std::floating_point U>
constexpr intervals::interval<T>
widenAdditively(intervals::interval<T> const& x, U dx)
{
    gsl_ExpectsDebug(dx >= 0);

    return intervals::interval<T>{ x.lower() - dx, x.upper_unchecked() + dx };
}


struct LogBucketingParams
{
    double bmin;      // minimum number of bins per unity interval
    double bmax;      // maximum number of bins per unity interval
    double xmin = 0;  // lower bound
    double x0 = 1;    // value around which to increase number of bins
    double dldx = 0;  // width of log-space interval over which to linearly increase number of bins
};

class LogBucketing
{
private:
    LogBucketingParams params_;
    double alpha_;  // = (b⁺ - b⁻)/(ξ⁺ - ξ⁻)  with  ξ = log x/x₀
    double beta_;  // = (ξ⁺⋅b⁻ - ξ⁻⋅b⁺)/(b⁺ - b⁻)  with  ξ = log x/x₀

    template <intervals::floating_point_interval_arg T>
    T
    bucketDensity(T x) const
    {
        using namespace intervals::math;
        using namespace intervals::logic;

        constexpr double rcpLog10 = 1./std::numbers::ln10;

        auto xb = max(x, params_.xmin);
        auto ldx = rcpLog10*log(xb/params_.x0);
        T result;
        double ldxmin = -0.5*params_.dldx;
        double ldxmax = +0.5*params_.dldx;
        auto below = ldx <= ldxmin;
        if (possibly(below))
        {
            auto b = params_.bmin;
            assign_partial(result, b);
        }
        auto above = ldx > ldxmax;
        if (possibly(above))
        {
            auto b = params_.bmax;
            assign_partial(result, b);
        }
        if (ldxmin != ldxmax)
        {
            auto linear = !below & !above;
            if (possibly(linear))
            {
                auto ldxc = constrain(ldx, linear);
                auto b = params_.bmin + alpha_*(ldxc - ldxmin);
                assign_partial(result, b);
            }
        }
        return result;
    }

public:
    LogBucketing(LogBucketingParams const& _params)
        : params_(_params)
    {
        gsl_Expects(params_.bmin > 0);
        gsl_Expects(params_.bmax > 0);
        gsl_Expects(params_.xmin >= 0);
        gsl_Expects(params_.x0 > 0);
        gsl_Expects(params_.dldx >= 0);

        double ldxmin = -0.5*params_.dldx;
        double ldxmax = +0.5*params_.dldx;
        if (ldxmin != ldxmax)
        {
            alpha_ = (params_.bmax - params_.bmin)/(ldxmax - ldxmin);
        }
        else
        {
            alpha_ = 0.;
        }
        if (params_.bmin != params_.bmax)
        {
            beta_ = (ldxmax*params_.bmin - ldxmin*params_.bmax)/(params_.bmax - params_.bmin);
        }
        else
        {
            beta_ = 0.;
        }
    }

    LogBucketingParams const&
    params() const
    {
        return params_;
    }

    template <intervals::floating_point_interval_arg T>
    auto
    map(T x) const
    {
        using namespace intervals::math;
        using namespace intervals::logic;

        auto xb = max(x, params_.xmin);
        auto ldx = log10(xb/params_.x0);
        auto result = T{ };
        double ldxmin = -0.5*params_.dldx;
        double ldxmax = +0.5*params_.dldx;
        auto below = ldx <= ldxmin;
        if (possibly(below))
        {
            auto ldxc = constrain(ldx, below);
            auto bldx = params_.bmin*ldxc;
            assign_partial(result, bldx);
        }
        auto above = ldx > ldxmax;
        if (possibly(above))
        {
            auto ldxc = constrain(ldx, above);
            auto bldx = params_.bmax*ldxc;
            assign_partial(result, bldx);
        }
        if (ldxmin != ldxmax)
        {
            auto linear = !below & !above;
            if (possibly(linear))
            {
                auto ldxc = constrain(ldx, linear);
                auto bldx = alpha_*(square(ldxc + beta_) - square(beta_));
                assign_partial(result, bldx);
            }
        }
        return narrow_cast<int>(floor(result));
    }

    template <intervals::integral_interval_arg T>
    auto
    fill(T label) const
    {
        gsl_Expects(params_.bmax == params_.bmin && "filling is supported only for constant bucketing density");

        using namespace intervals::math;

        auto b = params_.bmin;
        auto xflo = params_.x0*std::exp(std::numbers::ln10*infimum(label)/b);
        auto xfhi = params_.x0*std::exp(std::numbers::ln10*(supremum(label) + 1)/b);
        return intervals::interval{ xflo, xfhi };
    }

    template <intervals::floating_point_interval_arg T>
    auto
    widen(T x, double frac) const
    {
        using namespace intervals::math;

        auto b = bucketDensity(x);
        auto [blo, bhi] = intervals::interval(b);
        auto factor_lo = exp(-std::numbers::ln10*frac/blo);
        auto factor_hi = exp( std::numbers::ln10*frac/bhi);
        auto [xlo, xhi] = intervals::interval(x);
        return intervals::interval{ factor_lo*xlo, factor_hi*xhi };
    }
};


} // namespace rpmc


#endif // INCLUDED_RPMC_OPERATORS_RPMC_BUCKETS_LOG_HPP_
