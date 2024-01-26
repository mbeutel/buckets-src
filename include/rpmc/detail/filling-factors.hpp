
#ifndef INCLUDED_RPMC_DETAIL_FILLINGFACTORS_HPP_
#define INCLUDED_RPMC_DETAIL_FILLINGFACTORS_HPP_


#include <cmath>      // for abs()
#include <utility>    // for swap()
#include <algorithm>  // for min(), max()

#include <gsl-lite/gsl-lite.hpp>  // for gsl_ExpectsDebug()

#include <intervals/math.hpp>      // for square()
#include <intervals/logic.hpp>
#include <intervals/interval.hpp>


namespace rpmc {

namespace gsl = gsl_lite;

namespace detail {


inline double
XiNN(double h1, double h2, double da, double b)
{
    using namespace intervals::math;

    gsl_ExpectsDebug(h1 >= 0);
    gsl_ExpectsDebug(h2 >= h1);

    double b1 = da - h1 - h2;
    double b2 = da + h1 - h2;
    double b3 = da - h1 + h2;
    double b4 = da + h1 + h2;
    double xi0 = 4*h1*h2;
    if (b < b1)
    {
        return 0.;
    }
    else if (b < b2)
    {
        return 0.5*square(b - b1);
    }
    else if (b < b3)
    {
        return 2*h1*(b - b2 + h1);
    }
    else if (b < b4)
    {
        return xi0 - 0.5*square(b4 - b);
    }
    else
    {
        return xi0;
    }
}

inline double
fillingFactor(double h1, double h2, double a1, double a2, double R0, double R)
{
    gsl_ExpectsDebug(h1 >= 0);
    gsl_ExpectsDebug(h2 >= 0);
    gsl_ExpectsDebug(R0 >= 0);
    gsl_ExpectsDebug(R >= R0);

    if (h2 < h1)
    {
        using std::swap;
        swap(a1, a2);
        swap(h1, h2);
    }

    double da = a2 - a1;
    double xi0 = 4*h1*h2;
    return (XiNN(h1, h2, da, R) - XiNN(h1, h2, da, R0) + XiNN(h1, h2, da, -R0) - XiNN(h1, h2, da, -R))/xi0;
}
inline double
fillingFactor(double h1, double h2, double a1, double a2, double R)
{
    gsl_ExpectsDebug(h1 >= 0);
    gsl_ExpectsDebug(h2 >= 0);
    gsl_ExpectsDebug(R >= 0);

    if (h2 < h1)
    {
        using std::swap;
        swap(a1, a2);
        swap(h1, h2);
    }

    double da = a2 - a1;
    double xi0 = 4*h1*h2;
    return (XiNN(h1, h2, da, R) - XiNN(h1, h2, da, -R))/xi0;
}
inline double
fillingFactor(double h1, double h2, double R)
{
    return fillingFactor(h1, h2, 0., 0., R);
}


inline intervals::interval<double>
fillingFactor(
    intervals::interval<double> h1, intervals::interval<double> h2,
    intervals::interval<double> a1, intervals::interval<double> a2,
    intervals::interval<double> R)
{
    using namespace intervals::logic;
    using namespace intervals::math;

    gsl_ExpectsDebug(always(h1 > 0));
    gsl_ExpectsDebug(always(h2 > 0));
    gsl_ExpectsDebug(always(R >= 0));

    auto hmin = min(h1, h2);
    auto hmax = max(h1, h2);
    auto dh = max(0., hmax - hmin);
    auto qh_2 = 1./2*min(1., hmin/hmax);

    auto da = a2 - a1;
    auto absda = abs(da);
    auto brelp = R - da;
    auto brelm = -R - da;

        // Take some shortcuts.
    auto h = h1 + h2;
    if (always(absda >= R + h))
    {
            // We are definitely out of reach.
        return 0.;
    }
    else if (always(R >= h + absda))
    {
            // The interaction radius is large enough that the fill area will always be fully enclosed.
        return 1.;
    }
    else if (always(R < hmax) && possibly((brelm <= -dh) & (brelp >= dh)))
    {
            // Take advantage of  R ≪ h₁ + h₂ , in which case the filling factor is  ≪ 1 , and take the shortcut if the
            // central region is covered completely by  Δa  because then the upper limit wouldn't get any better than this
            // approximation.
        auto q = R/hmax;
        return intervals::interval{ 0., q.upper() };
    }

    auto b1 = -h1 - h2;
    auto b2 = -dh;
    auto b3 = dh;
    auto b4 = h1 + h2;

    auto rm = intervals::interval<double>{ };
    if (possibly(brelm < b1))
    {
        auto rb1m = 0.;
        assign_partial(rm, rb1m);
    }
    auto cm12 = (b1 <= brelm) & (brelm < b2);
    if (possibly(cm12))
    {
        auto brelmc = constrain(brelm, cm12);
        auto qbrelm = 1./2*(1 + (brelmc + hmax)/hmin);
        auto qbrelmc = max(0., min(1., qbrelm));

        auto r12m = qh_2*square(qbrelmc);
        assign_partial(rm, r12m);
    }
    auto cm23 = (b2 <= brelm) & (brelm < b3);
    if (possibly(cm23))
    {
        auto brelmc = constrain(brelm, cm23);
        auto qbrelm = 1./2*(1 + brelmc/hmax);
        auto qbrelmc = max(qh_2, min(1. - qh_2, qbrelm));

        auto r23m = qbrelmc;
        assign_partial(rm, r23m);
    }
    auto cm34 = (b3 <= brelm) & (brelm < b4);
    if (possibly(cm34))
    {
        auto brelmc = constrain(brelm, cm34);
        auto qbrelm = 1./2*((brelmc - hmax)/hmin - 1);
        auto qbrelmc = max(-1., min(0., qbrelm));

        auto r34m = 1. - qh_2*square(qbrelmc);
        assign_partial(rm, r34m);
    }
    if (possibly(b4 <= brelm))
    {
        auto ra4m = 1.;
        assign_partial(rm, ra4m);
    }

    auto result = intervals::interval<double>{ };
    if (possibly(brelp < b1))
    {
        assign_partial(result, 0.);
    }
    auto cp12 = (b1 <= brelp) & (brelp < b2);
    if (possibly(cp12))
    {
        auto brelpc = constrain(brelp, cp12);
        auto qbrelp = 1./2*(1 + (brelpc + hmax)/hmin);
        auto qbrelpc = max(0., min(1., qbrelp));

        if (possibly(brelm < b1))
        {
            auto r1 = qh_2*square(qbrelpc);
            assign_partial(result, r1);
        }
        auto cm1p = (b1 <= brelm) & (brelm <= brelp);
        if (possibly(cm1p))
        {
            auto brelmc = constrain(brelm, cm1p);
            auto qbrelm = 1./2*(1 + (brelmc + hmax)/hmin);
            auto qbrelmc = max(0., min(1., qbrelm));

            auto r2 = qh_2*max(0., square(qbrelpc) - square(qbrelmc));
            assign_partial(result, r2);
        }
    }
    auto cp23 = (b2 <= brelp) & (brelp < b3);
    if (possibly(cp23))
    {
        auto brelpc = constrain(brelp, cp23);
        auto qbrelp = 1./2*(1 + brelpc/hmax);
        auto qbrelpc = max(qh_2, min(1. - qh_2, qbrelp));

        if (possibly(brelm < b2))
        {
            auto rb2 = max(0., qbrelpc - rm);
            assign_partial(result, rb2);
        }
        if (possibly(b2 <= brelm))
        {
            //auto Rc = constrain_from_above(R, min(da, -da) + dh);
            ////auto Rc = min(R, min(da, -da) + dh);
            auto minda = min(0., min(da, -da));
            auto cRsmall = R <= minda + dh;
            if (possibly(cRsmall))
            {
                auto Rc = constrain(R, cRsmall);
                auto r23 = Rc/hmax;
                assign_partial(result, r23);
            }
        }
    }
    auto cp34 = (b3 <= brelp) & (brelp < b4);
    if (possibly(cp34))
    {
        auto brelpc = constrain(brelp, cp34);
        auto qbrelp = 1./2*((brelpc - hmax)/hmin - 1);
        auto qbrelpc = max(-1., min(0., qbrelp));

        if (possibly(brelm < b3))
        {
            auto rb3 = max(0., 1. - qh_2*square(qbrelpc) - rm);
            assign_partial(result, rb3);
        }
        auto cp3m = (b3 <= brelm) & (brelm <= brelp);
        if (possibly(cp3m))
        {
            auto brelmc = constrain(brelm, cp3m);
            auto qbrelm = 1./2*((brelmc - hmax)/hmin - 1);
            auto qbrelmc = max(-1., min(qbrelpc, qbrelm));

            auto r34 = qh_2*max(0., square(qbrelmc) - square(qbrelpc));
            assign_partial(result, r34);
        }
    }
    if (possibly(b4 <= brelp))
    {
        auto ra4 = 1. - rm;
        assign_partial(result, ra4);
    }

        // Take advantage of the situation where  R ≪ h₁ + h₂  with arbitrary  Δa . In this case the filling factor is
        //  ≪ 1 , and above computation doesn't do it justice because the arbitrarily large value of  Δa  makes it consider
        // different overlapping cases. (We might be able to improve this, but let's not.) We thus introduce an artificial
        // upper bound.
    auto q = min(R/hmax, 1.);
    return intervals::interval{ std::min(q.upper(), result.lower()), std::min(q.upper(), result.upper()) };
}
inline intervals::interval<double>
fillingFactor(
    intervals::interval<double> h1, intervals::interval<double> h2,
    intervals::interval<double> a1, intervals::interval<double> a2,
    intervals::interval<double> R0, intervals::interval<double> R)
{
    using namespace intervals::math;

    auto result = detail::fillingFactor(h1, h2, a1, a2, R) - detail::fillingFactor(h1, h2, a1, a2, R0);
    //return max(0., result);
    return constrain(result, result >= 0);
}
inline intervals::interval<double>
fillingFactor(
    intervals::interval<double> h1, intervals::interval<double> h2,
    intervals::interval<double> R)
{
    return detail::fillingFactor(h1, h2, intervals::interval(0.), intervals::interval(0.), R);
}


} // namespace detail

} // namespace rpmc


#endif // INCLUDED_RPMC_DETAIL_FILLINGFACTORS_HPP_
