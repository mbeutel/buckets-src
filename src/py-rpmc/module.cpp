
#include <pybind11/pybind11.h>


    // in py-rpmc-simulation.cpp
void
registerBindings_RPMCSimulation(pybind11::module m);

    // in py-rpmc-traditional-simulation.cpp
void
registerBindings_RPMCTraditionalSimulation(pybind11::module m);


PYBIND11_MODULE(rpmc, m)  // the first argument is the module name seen by Python
{
    registerBindings_RPMCSimulation(m);
    registerBindings_RPMCTraditionalSimulation(m);
}
