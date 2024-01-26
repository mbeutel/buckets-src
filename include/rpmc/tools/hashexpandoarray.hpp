
#ifndef INCLUDED_RPMC_TOOLS_HASHEXPANDOARRAY_HPP_
#define INCLUDED_RPMC_TOOLS_HASHEXPANDOARRAY_HPP_


#include <utility>        // for move()
#include <functional>     // for hash<>
#include <type_traits>    // for is_default_constructible<>
#include <unordered_map>

#include <gsl-lite/gsl-lite.hpp>  // for gsl_Assert()


namespace rpmc {

namespace gsl = ::gsl_lite;


template <typename T, typename MultiIndexT, typename HashT = std::hash<MultiIndexT>>
class HashExpandoArray
{
private:
    std::unordered_map<MultiIndexT, T, HashT> data_;

public:
    HashExpandoArray()
        : data_{ }
    {
    }

    T const&
    operator [](MultiIndexT const& index) const
    {
        auto it = data_.find(index);
        gsl_Assert(it != data_.end());
        return it->second;
    }
    T&
    operator [](MultiIndexT const& index)
    {
        auto it = data_.find(index);
        gsl_Assert(it != data_.end());
        return it->second;
    }

    T&
    obtain(MultiIndexT const& index)
    {
        return data_[index];
    }
    T&
    assign(MultiIndexT const& index, T value)
    {
        return data_.insert_or_assign(index, std::move(value)).first->second;
    }

    void
    remove(MultiIndexT const& index)
    {
        data_.erase(index);
    }
};


} // namespace rpmc


#endif // INCLUDED_RPMC_TOOLS_HASHEXPANDOARRAY_HPP_
