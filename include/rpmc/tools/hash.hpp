
// Common mathematical constants and functions.


#ifndef INCLUDED_RPMC_TOOLS_HASH_HPP_
#define INCLUDED_RPMC_TOOLS_HASH_HPP_


#include <climits>     // for CHAR_BIT
#include <cstdint>     // for uint[32|64]_t
#include <functional>  // for hash<>


namespace rpmc {

namespace detail {


// Hash-related code was borrowed from Boost.ContainerHash.
//
// Copyright 2022 Peter Dimov
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt


template <std::size_t Bits>
struct hash_mix_impl;

template <>
struct hash_mix_impl<64>
{
    static constexpr inline std::uint64_t
    fn(std::uint64_t x)
    {
        std::uint64_t const m = (std::uint64_t(0xe9846af) << 32) + 0x9b1a615d;

        x ^= x >> 32;
        x *= m;
        x ^= x >> 32;
        x *= m;
        x ^= x >> 28;

        return x;
    }
};

    // hash_mix for 32 bit size_t
    //
    // We use the "best xmxmx" implementation from
    // https://github.com/skeeto/hash-prospector/issues/19
template <>
struct hash_mix_impl<32>
{
    static constexpr inline std::uint32_t
    fn(std::uint32_t x)
    {
        std::uint32_t const m1 = 0x21f0aaad;
        std::uint32_t const m2 = 0x735a2d97;

        x ^= x >> 16;
        x *= m1;
        x ^= x >> 15;
        x *= m2;
        x ^= x >> 15;

        return x;
    }
};

constexpr inline std::size_t
hash_mix(std::size_t v)
{
    return hash_mix_impl<sizeof(std::size_t)*CHAR_BIT>::fn(v);
}

constexpr inline void
hash_combine_old_raw(std::size_t& seed, std::size_t hashval)
{
    seed ^= hashval + 0x9e3779b9 + (seed<<6) + (seed>>2);
}
template <typename T>
constexpr inline void
hash_combine_old(std::size_t& seed, T const& v)
{
    detail::hash_combine_old_raw(seed, std::hash<T>{ }(v));
}

constexpr inline void
hash_combine_raw(std::size_t& seed, std::size_t hashval)
{
    seed = detail::hash_mix(seed + 0x9e3779b9 + hashval);
}
template <typename T>
constexpr inline void
hash_combine(std::size_t& seed, T const& v)
{
    detail::hash_combine_raw(seed, std::hash<T>{ }(v));
}


} // namespace detail

} // namespace rpmc


#endif // INCLUDED_RPMC_TOOLS_HASH_HPP_
