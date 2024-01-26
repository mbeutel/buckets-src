
// ??


#ifndef INCLUDED_RPMC_TOOLS_SOA_HPP_
#define INCLUDED_RPMC_TOOLS_SOA_HPP_


#include <limits>
#include <concepts>     // for integral<>
#include <algorithm>    // for min(), max()
#include <type_traits>  // for is_signed<>

#include <makeshift/tuple.hpp>     // for value_tuple<>, template_for()
#include <makeshift/constval.hpp>
#include <makeshift/metadata.hpp>

#include <gsl-lite/gsl-lite.hpp>  // for gsl_Expects(), gsl_Assert()


namespace rpmc {

namespace gsl = gsl_lite;

namespace detail {


// TODO: if these are in the detail namespace, should this be a detail header?


template <template <template <typename> class TT> class DataT, template <typename> class SpanT>
constexpr gsl::dim
getSoALength(DataT<SpanT> const& data)
{
    using std::get;
    constexpr auto membersC = MAKESHIFT_CONSTVAL(makeshift::metadata::members<makeshift::value_tuple, DataT<SpanT>>());
    constexpr auto member0 = get<0>(membersC());
    return std::ssize(data.*member0);
}

template <template <template <typename> class TT> class DataT, template <typename> class SpanT>
constexpr void
assertSameSoALengths(DataT<SpanT> const& data)
{
    using std::get;

    auto length = detail::getSoALength(data);
    makeshift::apply(
        [length]
        (auto const&... members)
        {
            gsl_Assert(((std::ssize(members) == length) && ...));
        },
        makeshift::tie_members(data));
}

template <template <template <typename> class TT> class DataT, template <typename> class SpanT>
constexpr DataT<std::type_identity_t>
gatherFromSoA(DataT<SpanT> const& data, gsl::index i)
{
    return makeshift::apply(
        [i]
        (auto&&... members)
        {
            return DataT<std::type_identity_t>{ members[i]... };
        },
        makeshift::tie_members(data));
}


} // namespace detail

} // namespace rpmc


#endif // INCLUDED_RPMC_TOOLS_SOA_HPP_
