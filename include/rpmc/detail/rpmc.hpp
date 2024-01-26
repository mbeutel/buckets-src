
#ifndef INCLUDED_RPMC_DETAIL_RPMC_HPP_
#define INCLUDED_RPMC_DETAIL_RPMC_HPP_


#include <array>
#include <cstddef>   // for size_t
#include <optional>

#include <gsl-lite/gsl-lite.hpp>  // for index, gsl_Expects()

#include <makeshift/constval.hpp>

#include <rpmc/operators/rpmc/common.hpp>  // for DefaultLocator


namespace rpmc {

namespace gsl = gsl_lite;

namespace detail {


template <typename Is, template <gsl::index, template <typename> class> class ParticlePropertiesT, template <typename> class TT>
struct ParticlePropertiesTuple_;
template <std::size_t... Is, template <gsl::index, template <typename> class> class ParticlePropertiesT, template <typename> class TT>
struct ParticlePropertiesTuple_<std::index_sequence<Is...>, ParticlePropertiesT, TT>
{
    using type = std::tuple<ParticlePropertiesT<Is, TT>...>;
};

template <typename T, T N>
constexpr std::array<T, N>
array_iota()
{
    auto result = std::array<T, N>{ };
    for (T i = 0; i != N; ++i)
    {
        result[i] = i;
    }
    return result;
}

template <typename T>
constexpr bool
hasValue(T const&)
{
    return true;
}
template <typename T>
constexpr bool
hasValue(std::optional<T> const& maybeValue)
{
    return maybeValue.has_value();
}

template <typename T>
constexpr T&
getValue(T& value)
{
    return value;
}
template <typename T>
constexpr T&
getValue(std::optional<T>& maybeValue)
{
    return *maybeValue;
}
template <typename T>
constexpr T const&
getValue(std::optional<T> const& maybeValue)
{
    return *maybeValue;
}

template <typename T>
constexpr T&
getValueChecked(T& value)
{
    return value;
}
template <typename T>
constexpr T&
getValueChecked(std::optional<T>& maybeValue)
{
    return maybeValue.value();
}
template <typename T>
constexpr T const&
getValueChecked(std::optional<T> const& maybeValue)
{
    return maybeValue.value();
}

template <typename T>
constexpr T&
getOrInitValue(T& value)
{
    return value;
}
template <typename T>
constexpr T&
getOrInitValue(std::optional<T>& maybeValue)
{
    if (!maybeValue.has_value())
    {
        maybeValue.emplace();
    }
    return *maybeValue;
}

template <typename TTT, template <typename> class TT> struct Rebind;
template <template <template <typename> class> class TTT, template <typename> class TU, template <typename> class TT> struct Rebind<TTT<TU>, TT> { using type = TTT<TT>; };
template <template <template <typename> class> class TTT, template <typename> class TU, template <typename> class TT> struct Rebind<std::optional<TTT<TU>>, TT> { using type = std::optional<TTT<TT>>; };

//template <typename T> struct UnwrapOptional { using type = T; };
//template <typename T> struct UnwrapOptional<std::optional<T>> { using type = T; };



template <typename T, std::size_t N, typename TuplesT, std::size_t... Is>
constexpr std::array<T, sizeof...(Is)>
intersperseArraysImpl(TuplesT const& tuples, std::index_sequence<Is...>)
{
    constexpr std::size_t M = std::tuple_size_v<TuplesT>;
    using std::get;
    return { get<Is/M>(get<Is%M>(tuples))... };
}
void
getHomogeneousTupleSize() = delete; // cannot extract homogeneous tuple size from no tuples
template <typename Tuple0T, typename... TuplesT>
constexpr std::integral_constant<std::size_t, std::tuple_size_v<Tuple0T>>
getHomogeneousTupleSize(Tuple0T const&, TuplesT const&...)
{
    static_assert(((std::tuple_size_v<TuplesT> == std::tuple_size_v<Tuple0T>) && ...));
    return { };
}
template <typename T, typename... TuplesT>
constexpr auto
intersperseArrays(TuplesT const&... tuples)
{
    constexpr std::size_t n = decltype(detail::getHomogeneousTupleSize(tuples...))::value;
    return detail::intersperseArraysImpl<T, n>(std::tuple{ tuples... }, std::make_index_sequence<n*sizeof...(TuplesT)>{ });
}


template <typename ClassifierT, typename InteractionRangeT>
struct SubBucketLabelType
{
    using type = decltype(std::declval<ClassifierT>().subclassify(std::declval<InteractionRangeT>(), std::declval<ClassifierT>().location(gsl::index(0))));
};

template <typename T, typename... ArgsT>
constexpr void
destroyAndConstructInPlace(T& x, ArgsT&&... args) noexcept  // use `noexcept` to enforce termination if anything goes wrong
{
    x.~T();
    new (&x) T(std::forward<ArgsT>(args)...);
}

template <typename... InteractionModelsT>
constexpr std::array<gsl::index, sizeof...(InteractionModelsT)>
modelInteractionDistanceIndices()
{
    constexpr std::array haveLocality = { !std::is_same_v<typename InteractionModelsT::Locator, DefaultLocator>... };
    auto result = std::array<gsl::index, sizeof...(InteractionModelsT)>{ };
    gsl::index j = 0;
    for (gsl::index i = 0; i < gsl::dim(sizeof...(InteractionModelsT)); ++i)
    {
        if (haveLocality[i])
        {
            result[i] = j;
            ++j;
        }
        else
        {
            result[i] = -1;
        }
    }
    return result;
}


template <typename KeyTupleT, typename ValTupleT, std::size_t... Is>
constexpr std::array<gsl::index, std::tuple_size_v<KeyTupleT>>
firstTupleElementIndicesImpl(std::index_sequence<Is...>)
{
    return { makeshift::first_tuple_element_index_v<std::tuple_element_t<Is, KeyTupleT>, ValTupleT>... };
}
template <typename KeyTupleT, typename ValTupleT>
constexpr std::array<gsl::index, std::tuple_size_v<KeyTupleT>>
firstTupleElementIndices()
{
    return detail::firstTupleElementIndicesImpl<KeyTupleT, ValTupleT>(std::make_index_sequence<std::tuple_size_v<KeyTupleT>>{ });
}

template <template <typename...> class SeqT, typename... Ts, typename IndexT, IndexT... Is>
std::type_identity<SeqT<makeshift::nth_type_t<Is, Ts...>...>>
gatherSequence(std::type_identity<SeqT<Ts...>>, makeshift::array_constant<IndexT, Is...>)
{
    return { };
}
template <typename SeqT, typename IndicesT> struct GatherSequence_;
template <template <typename...> class SeqT, typename... Ts, typename IndexT, IndexT... Is>
struct GatherSequence_<SeqT<Ts...>, makeshift::array_constant<IndexT, Is...>>
    : decltype(detail::gatherSequence(std::type_identity<SeqT<Ts...>>{ }, makeshift::array_constant<IndexT, Is...>{ }))
{
};
template <typename SeqT, typename IndicesT> using GatherSequence = typename GatherSequence_<SeqT, IndicesT>::type;


} // namespace detail

} // namespace rpmc


#endif // INCLUDED_RPMC_DETAIL_RPMC_HPP_
