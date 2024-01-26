
#include <pybind11/pybind11.h>

#include <rpmc/operators/rpmc/buckets/log.hpp>

#include "py-utility.hpp"
#include "py-convert.hpp"


namespace py_rpmc {

namespace py = pybind11;


rpmc::LogBucketingParams
toLogBucketingParams(NamedObject<> obj)
{
    return rpmc::LogBucketingParams{
        .bmin = py_rpmc::castAttr<double>(obj, { "bmin" }),
        .bmax = py_rpmc::castAttr<double>(obj, { "bmax" }),
        .xmin = py_rpmc::castAttr<double>(obj, { "xmin" }),
        .x0 = py_rpmc::castAttr<double>(obj, { "x0" }),
        .dldx = py_rpmc::castAttr<double>(obj, { "dldx" })
    };
}


} // namespace py_rpmc
