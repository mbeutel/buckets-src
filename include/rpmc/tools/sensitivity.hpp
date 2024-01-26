
// Data types for sensitivity tracking.


#ifndef INCLUDED_RPMC_TOOLS_SENSITIVITY_HPP_
#define INCLUDED_RPMC_TOOLS_SENSITIVITY_HPP_


#include <cmath>
#include <array>

#include <intervals/math.hpp>  // for square()
#include <intervals/sign.hpp>

#include <gsl-lite/gsl-lite.hpp>  // for gsl_Expects()


namespace rpmc {

namespace gsl = ::gsl_lite;


struct SDouble
{
    double v;
    double dv;

public:
    SDouble() = default;

        // Implicitly construct a tuple  (c,∂c/∂x) = (c,0)  from parameter-independent value  c .
    /*implicit*/ constexpr SDouble(double c) noexcept
        : v(c), dv(0.)
    {
    }

        // Explicitly construct a tuple  (v,∂v/∂x)  from the given values.
    explicit constexpr SDouble(double _v, double _dv) noexcept
        : v(_v), dv(_dv)
    {
    }

        // Construct the tuple  (x,∂x/∂x) = (x, 1)  from the parameter  x .
    static constexpr SDouble
    param(double x) noexcept
    {
        return SDouble{ x, 1. };
    }

        // Return the value  v .
    constexpr double
    value() const noexcept
    {
        return v;
    }

        // Return the parameter sensitivity  ∂v/∂x  of the value.
    constexpr double
    sensitivity() const noexcept
    {
        return dv;
    }

        // Define the usual arithmetic operations.
    [[nodiscard]] friend constexpr SDouble
    operator +(SDouble x) noexcept
    {
        return x;
    }
    [[nodiscard]] friend constexpr SDouble
    operator -(SDouble x) noexcept
    {
        return SDouble{
            -x.v,
            -x.dv  // -dx
        };
    }
    [[nodiscard]] friend constexpr SDouble
    operator +(SDouble x, SDouble y) noexcept
    {
        return SDouble{
            x.v + y.v,
            x.dv + y.dv  // dx + dy
        };
    }
    [[nodiscard]] friend constexpr SDouble
    operator -(SDouble x, SDouble y) noexcept
    {
        return SDouble{
            x.v - y.v,
            x.dv - y.dv  // dx - dy
        };
    }
    [[nodiscard]] friend constexpr SDouble
    operator *(SDouble x, SDouble y) noexcept
    {
        return SDouble{
            x.v*y.v,
            y.v*x.dv + x.v*y.dv  // y dx + x dy
        };
    }
    [[nodiscard]] friend constexpr SDouble
    operator /(SDouble x, SDouble y) noexcept
    {
        return SDouble{
            x.v/y.v,
            (y.v*x.dv - x.v*y.dv)/(y.v*y.v)  // (y dx - x dy)/y²
        };
    }
    constexpr SDouble&
    operator +=(SDouble rhs) noexcept
    {
        return *this = *this + rhs;
    }
    constexpr SDouble&
    operator -=(SDouble rhs) noexcept
    {
        return *this = *this - rhs;
    }
    constexpr SDouble&
    operator *=(SDouble rhs) noexcept
    {
        return *this = *this*rhs;
    }
    constexpr SDouble&
    operator /=(SDouble rhs) noexcept
    {
        return *this = *this/rhs;
    }

        // Define comparison operators.
    [[nodiscard]] friend constexpr bool
    operator ==(SDouble x, SDouble y) noexcept
    {
        return x.value() == y.value();
    }
    [[nodiscard]] friend constexpr bool
    operator !=(SDouble x, SDouble y) noexcept
    {
        return x.value() != y.value();
    }
    [[nodiscard]] friend constexpr bool
    operator <(SDouble x, SDouble y) noexcept
    {
        return x.value() < y.value();
    }
    [[nodiscard]] friend constexpr bool
    operator >(SDouble x, SDouble y) noexcept
    {
        return x.value() > y.value();
    }
    [[nodiscard]] friend constexpr bool
    operator <=(SDouble x, SDouble y) noexcept
    {
        return x.value() <= y.value();
    }
    [[nodiscard]] friend constexpr bool
    operator >=(SDouble x, SDouble y) noexcept
    {
        return x.value() >= y.value();
    }
    [[nodiscard]] friend constexpr bool
    operator ==(double x, SDouble y) noexcept
    {
        return x == y.value();
    }
    [[nodiscard]] friend constexpr bool
    operator !=(double x, SDouble y) noexcept
    {
        return x != y.value();
    }
    [[nodiscard]] friend constexpr bool
    operator <(double x, SDouble y) noexcept
    {
        return x < y.value();
    }
    [[nodiscard]] friend constexpr bool
    operator >(double x, SDouble y) noexcept
    {
        return x > y.value();
    }
    [[nodiscard]] friend constexpr bool
    operator <=(double x, SDouble y) noexcept
    {
        return x <= y.value();
    }
    [[nodiscard]] friend constexpr bool
    operator >=(double x, SDouble y) noexcept
    {
        return x >= y.value();
    }
    [[nodiscard]] friend constexpr bool
    operator ==(SDouble x, double y) noexcept
    {
        return x.value() == y;
    }
    [[nodiscard]] friend constexpr bool
    operator !=(SDouble x, double y) noexcept
    {
        return x.value() != y;
    }
    [[nodiscard]] friend constexpr bool
    operator <(SDouble x, double y) noexcept
    {
        return x.value() < y;
    }
    [[nodiscard]] friend constexpr bool
    operator >(SDouble x, double y) noexcept
    {
        return x.value() > y;
    }
    [[nodiscard]] friend constexpr bool
    operator <=(SDouble x, double y) noexcept
    {
        return x.value() <= y;
    }
    [[nodiscard]] friend constexpr bool
    operator >=(SDouble x, double y) noexcept
    {
        return x.value() >= y;
    }

        // Define indirect value accessors.
    [[nodiscard]] friend constexpr int
    sgn(SDouble x) noexcept
    {
        return intervals::sgn(x.v);
    }

        // Define combined arithmetic operations.
    [[nodiscard]] friend SDouble
    fabs(SDouble x) noexcept
    {
        return SDouble{
            std::fabs(x.v),
            int(intervals::sgn(x.v))*x.dv
        };
    }
    [[nodiscard]] friend constexpr SDouble
    square(SDouble x) noexcept
    {
        return SDouble{
            intervals::square(x.v),
            2*x.v*x.dv
        };
    }
    [[nodiscard]] friend constexpr SDouble
    cube(SDouble x) noexcept
    {
        return SDouble{
            intervals::cube(x.v),
            3*intervals::square(x.v)*x.dv
        };
    }

        // Define other arithmetic operations.
    [[nodiscard]] friend SDouble
    sqrt(SDouble x) noexcept
    {
        auto sqrtX = std::sqrt(x.v);
        return SDouble{
            sqrtX,
            x.dv/(2*sqrtX)  // 1/2 dx/√x
        };
    }
    [[nodiscard]] friend SDouble
    log(SDouble x) noexcept
    {
        return SDouble{
            std::log(x.v),
            x.dv/x.v  // dx/x
        };
    }
    [[nodiscard]] friend SDouble
    exp(SDouble x) noexcept
    {
        auto expX = std::exp(x.v);
        return SDouble{
            expX,
            expX*x.dv,  // exp(x) dx
        };
    }
    [[nodiscard]] friend SDouble
    pow(SDouble b, SDouble e) noexcept
    {
            // TODO: do we need to special-case  b = 0  here?
        auto logB = std::log(b.v);
        auto expElogB = std::exp(e.v*logB);
        return SDouble{
            expElogB,  // pow(b, e) = exp(e log b)
            expElogB*(logB*e.dv + e.v*b.dv/b.v)  // exp(e log b) d(e log b) = pow(b, e) [log b de + e db/b]
        };
    }

        // Define trigonometric operations.
    [[nodiscard]] friend SDouble
    sin(SDouble x) noexcept
    {
        return SDouble{
            std::sin(x.v),
            std::cos(x.v)*x.dv  // cos x dx
        };
    }
    [[nodiscard]] friend SDouble
    cos(SDouble x) noexcept
    {
        return SDouble{
            std::cos(x.v),
            -std::sin(x.v)*x.dv  // -sin x dx
        };
    }
    [[nodiscard]] friend SDouble
    tan(SDouble x) noexcept
    {
        auto tanX = std::tan(x.v);
        return SDouble{
            tanX,
            (1. + intervals::square(tanX))*x.dv  // (1 + tan² x) dx
        };
    }
    [[nodiscard]] friend SDouble
    asin(SDouble x) noexcept
    {
        return SDouble{
            std::asin(x.v),
            x.dv/std::sqrt(1 - intervals::square(x.v))  // dx/√(1 - x²)
        };
    }
    [[nodiscard]] friend SDouble
    acos(SDouble x) noexcept
    {
        return SDouble{
            std::acos(x.v),
            -x.dv/std::sqrt(1 - intervals::square(x.v))  // -dx/√(1 - x²)
        };
    }
    [[nodiscard]] friend SDouble
    atan(SDouble x) noexcept
    {
        return SDouble{
            std::atan(x.v),
            x.dv/(1. + intervals::square(x.v))  // dx/(1 + x²)
        };
    }
};


} // namespace rpmc


#endif // INCLUDED_RPMC_TOOLS_SENSITIVITY_HPP_
