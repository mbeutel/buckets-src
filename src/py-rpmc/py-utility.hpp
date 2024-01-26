
#ifndef INCLUDED_RPMC_PY_UTILITY_HPP_
#define INCLUDED_RPMC_PY_UTILITY_HPP_


#include <span>
#include <string>
#include <cstddef>           // for size_t
#include <utility>           // for move(), forward<>()
#include <exception>
#include <stdexcept>         // for runtime_error, invalid_argument
#include <string_view>
#include <initializer_list>

#include <gsl-lite/gsl-lite.hpp>  // for dim, index, narrow<>(), czstring

#include <fmt/core.h>  // for format()

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


namespace py_rpmc {

namespace py = pybind11;


template <typename T>
using py_array = py::array_t<T>;


template <typename T = py::object>
struct NamedObject
{
    T obj;
    std::string name;

    NamedObject(T _obj)
        : obj(std::forward<T>(_obj))
    {
    }
    NamedObject(T _obj, std::string_view _name)
        : obj(std::forward<T>(_obj)), name(_name)
    {
    }

    operator T(void) const
    {
        return obj;
    }
};


template <typename T, int Flags = py::array::forcecast>
NamedObject<py::array_t<T, Flags>>
expectArray(NamedObject<> obj, std::string_view name = { })
{
    if (!py::array::check_(obj.obj))
    {
        throw std::invalid_argument(fmt::format("{}expected 'numpy.ndarray' but got a {}",
            obj.name.empty() ? std::string{ } : '\'' + std::string(obj.name) + "': ", py::cast<std::string>(py::str(py::type::of(obj.obj)))));
    }
    if (!py::array_t<T>::check_(obj.obj))
    {
        throw std::invalid_argument(fmt::format("{}expected 'numpy.ndarray' of type {} but got a 'numpy.ndarray' of type {}",
            obj.name.empty() ? std::string{ } : '\'' + std::string(obj.name) + "': ", py::type_id<T>(), py::cast<std::string>(py::str(py::array(obj.obj).dtype()))));
    }
    if constexpr ((Flags & (py::array::c_style | py::array::f_style)) != 0)
    {
        if (!py::array_t<T, Flags>::check_(obj.obj))
        {
            std::string layout;
            if constexpr ((Flags & py::array::c_style) != 0)
            {
                layout = "C";
            }
            if constexpr ((Flags & py::array::f_style) != 0)
            {
                layout = "Fortran";
            }
            throw std::invalid_argument(fmt::format("{}expected 'numpy.ndarray' with {} layout",
                obj.name.empty() ? std::string{ } : '\'' + obj.name + "': ", layout));
        }
    }
    auto result = py::array_t<T, Flags>(obj.obj);
    gsl_Assert(result.is(obj.obj));
    return { result, name };
}


template <typename T, int Flags>
std::span<T>
_asSpanImpl(NamedObject<py::array_t<std::remove_const_t<T>, Flags>> array)
{
    if (array.obj.ndim() != 1)
    {
        throw std::invalid_argument(fmt::format("{}array must have dimension 1",
            array.name.empty() ? std::string{ } : '\'' + array.name + "': "));
    }
    if (array.obj.strides(0) != sizeof(T))
    {
        throw std::invalid_argument(fmt::format("{}array must be contiguous",
            array.name.empty() ? std::string{ } : '\'' + array.name + "': "));
    }
    auto size = gsl_lite::narrow<std::size_t>(array.obj.shape(0));
    return { array.obj.mutable_data(), size };
}
template <typename T, int Flags>
requires std::is_const_v<T>
std::span<T>
asSpan(NamedObject<py::array_t<std::remove_const_t<T>, Flags>> array)
{
    return py_rpmc::_asSpanImpl<T, Flags>(array);
}
template <typename T, int Flags>
std::span<T>
asSpan(NamedObject<py::array_t<T, Flags>> array)
{
    return py_rpmc::_asSpanImpl<T, Flags>(array);
}

template <typename T, int Flags>
std::span<T>
asRaveledSpan(NamedObject<> obj)
{
    static_assert((Flags & (py::array::c_style | py::array::f_style)) != 0, "need to specify array order");

    auto array = py_rpmc::expectArray<std::remove_const_t<T>, Flags>(obj);

    gsl_lite::dim stride = 1;
    if constexpr ((Flags & py::array::c_style) != 0)
    {
        for (gsl_lite::index i = array.obj.ndim() - 1; i >= 0; --i)
        {
            if (array.obj.shape(i) != 0 && array.obj.strides(i) != stride*gsl_lite::dim(sizeof(T)))
            {
                throw std::invalid_argument(fmt::format("{}array must be contiguous and in C order (axis {} of {}-dimensional array: expected stride {} but got {})",
                    array.name.empty() ? std::string{ } : '\'' + array.name + "': ", i, array.obj.ndim(), stride*gsl_lite::dim(sizeof(T)), array.obj.strides(i)));
            }
            stride *= array.obj.shape(i);
        }
    }
    else if constexpr ((Flags & py::array::f_style) != 0)
    {
        for (gsl_lite::index i = 0; i < array.obj.ndim(); ++i)
        {
            if (array.obj.shape(i) != 0 && array.obj.strides(i) != stride*gsl_lite::dim(sizeof(T)))
            {
                throw std::invalid_argument(fmt::format("{}array must be contiguous and in Fortran order (axis {} of {}-dimensional array: expected stride {} but got {})",
                    array.name.empty() ? std::string{ } : '\'' + array.name + "': ", i, array.obj.ndim(), stride*gsl_lite::dim(sizeof(T)), array.obj.strides(i)));
            }
            stride *= array.obj.shape(i);
        }
    }
    auto size = gsl_lite::narrow<std::size_t>(stride);
    return { array.obj.mutable_data(), size };
}


inline std::string
qualifiedName(std::string_view scope, std::string_view name)
{
    return scope.empty() ? std::string(name) : std::string(scope) + '.' + std::string(name);
}

template <typename T>
NamedObject<py::array_t<T>>
dataframeColumn(NamedObject<> df, gsl_lite::czstring colName)
{
    auto col = df.obj.attr(colName);
    auto dfColName = py_rpmc::qualifiedName(df.name, colName);
    if (!py::cast<py::dtype>(col.attr("dtype")).is(py::dtype::of<T>()))
    {
            // This also catches columns of type `object`.
        throw std::invalid_argument(fmt::format("'{}': expected column of type {} but got a {} of type {}",
            dfColName,
            py::type_id<T>(), py::cast<std::string>(py::str(py::type::of(col))), py::cast<std::string>(py::str(py::cast<py::dtype>(col.attr("dtype"))))));
    }
    return py_rpmc::expectArray<T>({ col.attr("to_numpy")(), dfColName });
}
template <typename T>
std::span<T>
dataframeColumnSpan(NamedObject<> df, gsl_lite::czstring colName)
{
    return py_rpmc::asSpan<T>(py_rpmc::dataframeColumn<T>(df, colName));
}


inline NamedObject<>
getAttr(NamedObject<> obj, std::span<std::string_view const> members)
{
    try
    {
        py::object lobj = obj.obj;
        for (std::string_view member : members)
        {
            obj.name += '.';
            obj.name += member;
            obj.obj = obj.obj.attr(std::string(member).c_str());
        }
        return obj;
    }
    catch (std::exception const& e)
    {
        std::throw_with_nested(std::runtime_error(fmt::format("'{}': {}", obj.name, e.what())));
    }
}
inline NamedObject<>
getAttr(NamedObject<> obj, std::initializer_list<std::string_view> members)
{
    return py_rpmc::getAttr(obj, std::span(members));
}

template <typename T>
T
cast(NamedObject<> obj)
{
    try
    {
        return py::cast<T>(obj.obj);
    }
    catch (std::exception const& e)
    {
        std::throw_with_nested(std::runtime_error(fmt::format("'{}': {}", obj.name, e.what())));
    }
}

template <typename T>
T
castAttr(NamedObject<> obj, std::span<std::string_view const> members)
{
    return py_rpmc::cast<T>(py_rpmc::getAttr(obj, members));
}
template <typename T>
T
castAttr(NamedObject<> obj, std::initializer_list<std::string_view> members)
{
    return py_rpmc::castAttr<T>(obj, std::span(members));
}


} // namespace py_rpmc


#endif // INCLUDED_RPMC_PY_UTILITY_HPP_
