
#include <array>
#include <tuple>
#include <string>
#include <random>       // for minstd_rand
#include <utility>      // for move()
#include <string_view>

#include <gsl-lite/gsl-lite.hpp>  // for dim, index

#include <fmt/core.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <makeshift/string.hpp>   // for parse_enum<>(), parse_flags<>()
#include <makeshift/variant.hpp>  // for visit(), variant_transform(), expand_failfast()

#include <rpmc/tools/utility.hpp>
#include <rpmc/tools/particles.hpp>

#include <rpmc/simulations/simulation.hpp>
#include <rpmc/simulations/event-driven.hpp>
#include <rpmc/operators/operator.hpp>                                // for InspectionData
#include <rpmc/operators/rpmc.hpp>
#include <rpmc/operators/rpmc/common.hpp>
#include <rpmc/operators/rpmc/models/stirring.hpp>
#include <rpmc/operators/rpmc/models/friction.hpp>
#include <rpmc/operators/rpmc/models/collision/test.hpp>
#include <rpmc/operators/rpmc/models/collision/geometric.hpp>
#include <rpmc/operators/rpmc/classifiers/mers.hpp>
#include <rpmc/operators/rpmc/classifiers/mass-and-swarm-regime.hpp>
#include <rpmc/operators/rpmc/locators/log.hpp>
#include <rpmc/operators/rpmc/locators/linear.hpp>

#include "pybind11ext.hpp"
#include "py-utility.hpp"
#include "py-convert.hpp"
#include "py-simulation.hpp"


void
registerBindings_RPMCSimulation(pybind11::module m)
{
    namespace gsl = gsl_lite;
    namespace py = pybind11;
    using namespace py_rpmc;
    using namespace std::literals;


    auto pyRPMCSimulation = pybind11ext::anonymous_class(m, "RPMCSimulation",
        R"doc()doc",
        py::init(
            [](py::object args, py::object state, bool /*log*/, bool /*display*/)
            {
                using RNG = std::minstd_rand;

                auto effects = makeshift::parse_flags<Effects>(castAttr<std::string>({ args, "args" }, { "simulation", "effects" }));
                auto options = makeshift::parse_flags<Options>(castAttr<std::string>({ args, "args" }, { "simulation", "options" }));
                auto bucketExhaustion = makeshift::parse_flags<rpmc::BucketExhaustion>(castAttr<std::string>({ args, "args" }, { "simulation", "bucket_exhaustion" }));

                auto M = dataframeColumnSpan<double>({ state, "state" }, "M");
                auto m = dataframeColumnSpan<double>({ state, "state" }, "m");
                auto N = dataframeColumnSpan<double>({ state, "state" }, "N");
                auto a = dataframeColumnSpan<double>({ state, "state" }, "a");
                auto e = dataframeColumnSpan<double>({ state, "state" }, "e");
                auto sininc = dataframeColumnSpan<double>({ state, "state" }, "sininc");
                auto rho = dataframeColumnSpan<double>({ state, "state" }, "ρ");
                auto St = dataframeColumnSpan<double>({ state, "state" }, "St");
                auto vr = dataframeColumnSpan<double>({ state, "state" }, "vr");
                auto vphi = dataframeColumnSpan<double>({ state, "state" }, "vφ");
                auto hd = dataframeColumnSpan<double>({ state, "state" }, "hd");

                auto particleData = rpmc::PParticleData(rpmc::SpanParticleData(
                    std::tuple{ }, // properties that appertain to particle identity
                    std::tuple{ M, m, N, a, e, sininc, rho, St, vr, vphi, hd }));  // all other properties

                auto localityV = makeshift::expand_failfast((options & Options::locality) != Options::none);

                auto makeLinearTestLocator = [&]
                {
                    return rpmc::LinearLocator{
                        rpmc::LinearLocatorParams{
                            .referenceLocation = 0.,
                            .drMin = 1.,
                            .dsMin = castAttr<int>({ args, "args" }, { "simulation", "subclass_resolution_factor" }),
                        },
                        rpmc::LinearLocatorState{
                            .a = a
                        }
                    };
                };
                auto makeLogLocator = [&]
                {
                    return rpmc::LogLocator{
                        rpmc::LogLocatorParams{
                            .referenceLocation = castAttr<double>({ args, "args" }, { "ring", "r" }),
                            .drMin = castAttr<double>({ args, "args" }, { "zones", "ΔrMin" }),
                            .dsMin = castAttr<int>({ args, "args" }, { "simulation", "subclass_resolution_factor" }),
                        },
                        rpmc::LogLocatorState{
                            .a = a
                        }
                    };
                };

                auto makeMassClassifier = [&]
                {
                    return rpmc::ParticleMassAndSwarmRegimeRPMCClassifier{
                        rpmc::ParticleMassAndSwarmRegimeRPMCClassifierParams{
                            .referenceMass = 1.,  // (g)
                            .MBucketingParams = toLogBucketingParams(getAttr({ args, "args" }, { "simulation", "M_bins_per_decade" })),
                            .mBucketingParams = toLogBucketingParams(getAttr({ args, "args" }, { "simulation", "m_bins_per_decade" })),
                            .particleRegimeThreshold = castAttr<int>({ args, "args" }, { "simulation", "particle_regime_threshold" })
                        },
                        rpmc::ParticleMassAndSwarmRegimeRPMCClassifierState{
                            .M = M,
                            .m = m
                        }
                    };
                };
                auto makeGeometricClassifier = [&]
                {
                    return rpmc::ParticleMERSRPMCClassifier{
                        rpmc::ParticleMERSRPMCClassifierParams{
                            .MStar =  castAttr<double>({ args, "args" }, { "star", "M" }),
                            .referenceRadius = castAttr<double>({ args, "args" }, { "ring", "r" }),
                            .dr = castAttr<double>({ args, "args" }, { "ring", "Δr" })/castAttr<double>({ args, "args" }, { "simulation", "r_bins" }),
                            .binWideningFraction = castAttr<double>({ args, "args" }, { "simulation", "bin_widening" }),  // TODO: rename on Python side
                            .subclassWideningFraction = castAttr<double>({ args, "args" }, { "simulation", "subclass_widening_fraction" }),
                            .MBucketingParams = toLogBucketingParams(getAttr({ args, "args" }, { "simulation", "M_bins_per_decade" })),
                            .mBucketingParams = toLogBucketingParams(getAttr({ args, "args" }, { "simulation", "m_bins_per_decade" })),
                            .eBucketingParams = toLogBucketingParams(getAttr({ args, "args" }, { "simulation", "e_bins_per_decade" })),
                            .sinincBucketingParams = toLogBucketingParams(getAttr({ args, "args" }, { "simulation", "sininc_bins_per_decade" })),
                            .particleRegimeThreshold = castAttr<int>({ args, "args" }, { "simulation", "particle_regime_threshold" }),
                            .exhaustion = bucketExhaustion
                        },
                        rpmc::ParticleMERSRPMCClassifierState{
                            .M = M,
                            .m = m,
                            .N = N,
                            .a = a,
                            .e = e,
                            .sininc = sininc
                        }
                    };
                };
                auto makeCollisionOperator = [&]() -> rpmc::PDiscreteOperator
                {
                    auto collisionKernelStr = castAttr<std::string>({ args, "args" }, { "collisions", "kernel" });
                    auto collisionKernel = makeshift::parse_enum<CollisionKernel>(collisionKernelStr);
                    auto collisionKernelV = makeshift::expand_failfast(collisionKernel);

                    auto classifierLocatorCollisionModelV = makeshift::variant_transform(
                        [&](auto collisionKernelC, auto localityC)
                        {
                            [[maybe_unused]] auto getCollisionModelBaseParams = [&]
                            {
                                return rpmc::CollisionModelBaseParams{
                                    .relativeChangeRate = castAttr<double>({ args, "args" }, { "simulation", "mass_growth_factor" }),
                                    .particleRegimeThreshold = castAttr<int>({ args, "args" }, { "simulation", "particle_regime_threshold" }),
                                    .particleRegimeThresholdForInteractionRates = castAttr<double>({ args, "args" }, { "simulation", "particle_regime_threshold_for_interaction_rates" })
                                };
                            };
                            [[maybe_unused]] auto getCollisionModelBaseState = [&]
                            {
                                return rpmc::CollisionModelBaseState{
                                    .M = M,
                                    .m = m,
                                    .N = N
                                };
                            };

                            if constexpr (collisionKernelC == CollisionKernel::constant && !localityC)
                            {
                                return std::tuple{
                                    makeMassClassifier(),
                                    rpmc::DefaultLocator{ },
                                    rpmc::ConstantKernelModel{
                                        rpmc::ConstantKernelModelParams{
                                            getCollisionModelBaseParams(),
                                            /*.collisionRate =*/ castAttr<double>({ args, "args" }, { "collisions", "constant_collision_rate" })
                                        },
                                        getCollisionModelBaseState()
                                    }
                                };
                            }
                            else if constexpr (collisionKernelC == CollisionKernel::linear)
                            {
                                auto locator = rpmc::if_else_c(localityC, makeLinearTestLocator(), rpmc::DefaultLocator{ });
                                return std::tuple{
                                    makeMassClassifier(),
                                    locator,
                                    rpmc::LinearKernelModel<localityC>{
                                        rpmc::LinearKernelModelParams{
                                            getCollisionModelBaseParams(),
                                            /*.collisionRateCoefficient =*/ castAttr<double>({ args, "args" }, { "collisions", "linear_collision_rate_coefficient" })
                                        },
                                        getCollisionModelBaseState()
                                    }
                                };
                            }
                            else if constexpr (collisionKernelC == CollisionKernel::product)
                            {
                                return std::tuple{
                                    makeMassClassifier(),
                                    rpmc::DefaultLocator{ },
                                    rpmc::ProductKernelModel{
                                        rpmc::ProductKernelModelParams{
                                            getCollisionModelBaseParams(),
                                            /*.collisionRateCoefficient =*/ castAttr<double>({ args, "args" }, { "collisions", "product_collision_rate_coefficient" })
                                        },
                                        getCollisionModelBaseState()
                                    }
                                };
                            }
                            else if constexpr (collisionKernelC == CollisionKernel::constantThreshold)
                            {
                                return std::tuple{
                                    makeMassClassifier(),
                                    rpmc::DefaultLocator{ },
                                    rpmc::ConstantThresholdKernelModel{
                                        rpmc::ConstantThresholdKernelModelParams{
                                            getCollisionModelBaseParams(),
                                            /*.mThreshold =*/ castAttr<double>({ args, "args" }, { "collisions", "constant_threshold_mass" }),
                                            /*.collisionRate =*/ castAttr<double>({ args, "args" }, { "collisions", "constant_collision_rate" })
                                        },
                                        getCollisionModelBaseState()
                                    }
                                };
                            }
                            else if constexpr (collisionKernelC == CollisionKernel::linearThreshold)
                            {
                                return std::tuple{
                                    makeMassClassifier(),
                                    rpmc::DefaultLocator{ },
                                    rpmc::LinearThresholdKernelModel{
                                        rpmc::LinearThresholdKernelModelParams{
                                            getCollisionModelBaseParams(),
                                            /*.mThreshold =*/ castAttr<double>({ args, "args" }, { "collisions", "linear_threshold_mass" }),
                                            /*.collisionRateCoefficient =*/ castAttr<double>({ args, "args" }, { "collisions", "linear_collision_rate_coefficient" })
                                        },
                                        getCollisionModelBaseState()
                                    }
                                };
                            }
                            else if constexpr (collisionKernelC == CollisionKernel::runaway && !localityC)
                            {
                                return std::tuple{
                                    makeMassClassifier(),
                                    rpmc::DefaultLocator{ },
                                    rpmc::RunawayKernelModel{
                                        rpmc::RunawayKernelModelParams{
                                            getCollisionModelBaseParams(),
                                            /*.collisionRateCoefficient =*/ castAttr<double>({ args, "args" }, { "collisions", "runaway_collision_rate_coefficient" }),
                                            /*.criticalMass =*/ castAttr<double>({ args, "args" }, { "collisions", "runaway_critical_mass" })
                                        },
                                        getCollisionModelBaseState()
                                    }
                                };
                            }
                            else if constexpr (collisionKernelC == CollisionKernel::geometric)
                            {
                                auto locator = rpmc::if_else_c(localityC, makeLogLocator(), rpmc::DefaultLocator{ });
                                return std::tuple{
                                    makeGeometricClassifier(),
                                    locator,
                                    rpmc::GeometricCollisionModel{
                                        rpmc::GeometricCollisionModelParams{
                                            getCollisionModelBaseParams(),
                                            /*.MStar =*/ castAttr<double>({ args, "args" }, { "star", "M" }),
                                            /*.drMin =*/ castAttr<double>({ args, "args" }, { "zones", "ΔrMin" }),
                                            /*.restitutionCoef =*/ castAttr<double>({ args, "args" }, { "collisions", "ε" }),
                                            /*.fragmentRadius =*/ castAttr<double>({ args, "args" }, { "collisions", "Rfrag" }),
                                            /*.collisionOutcomes =*/ makeshift::parse_flags<rpmc::CollisionOutcomes>(castAttr<std::string>({ args, "args" }, { "collisions", "outcomes" })),
                                            /*.NThreshold =*/ castAttr<double>({ args, "args" }, { "simulation", "N_threshold" }),
                                            /*.StNBodyThreshold =*/ castAttr<double>({ args, "args" }, { "simulation", "St_NBody_threshold" }),
                                            /*.mNBodyThreshold =*/ castAttr<double>({ args, "args" }, { "simulation", "m_NBody_threshold" }),
                                            /*.StDustThreshold =*/ castAttr<double>({ args, "args" }, { "simulation", "St_dust_threshold" }),
                                            /*.mDustThreshold =*/ castAttr<double>({ args, "args" }, { "simulation", "m_dust_threshold" })
                                        },
                                        rpmc::GeometricCollisionModelState{
                                            getCollisionModelBaseState(),
                                            /*.a =*/ a,
                                            /*.e =*/ e,
                                            /*.sininc =*/ sininc,
                                            /*.vr =*/ vr,
                                            /*.vphi =*/ vphi,
                                            /*.hd =*/ hd,
                                            /*.St =*/ St,
                                            /*.rho =*/ rho
                                        }
                                    }
                                };
                            }
                            else
                            {
                                throw std::runtime_error(
                                    fmt::format(
                                        "collision kernel '{}' not supported in standalone RPMC simulation",
                                        collisionKernelStr));
                            }
                        },
                        collisionKernelV, localityV);
                    return makeshift::visit(
                        [&](auto&& classifierLocatorCollisionModel) -> rpmc::PDiscreteOperator
                        {
                            auto& [classifier, locator, collisionModel] = classifierLocatorCollisionModel;
                            return rpmc::RPMCOperator(
                                rpmc::RPMCOperatorParams{
                                    .removalBucketUpdateDelay = castAttr<double>({ args, "args" }, { "simulation", "removal_bucket_update_delay" }),
                                    .rejectionBucketUpdateDelay = castAttr<double>({ args, "args" }, { "simulation", "rejection_bucket_update_delay" })
                                },
                                particleData,
                                RNG(castAttr<unsigned int>({ args, "args" }, { "simulation", "random_seed" })),
                                classifier,
                                locator,
                                collisionModel);
                        },
                        classifierLocatorCollisionModelV);
                };
                auto collisionOperator = (effects & Effects::collisions) != Effects::none
                    ? makeCollisionOperator()
                    : rpmc::PDiscreteOperator(rpmc::NoOpDiscreteOperator{ });

                auto makeStirringOperator = [&]() -> rpmc::PDiscreteOperator
                {
                    return makeshift::visit(
                        [&](auto localityC) -> rpmc::PDiscreteOperator
                        {
                            auto classifier = makeGeometricClassifier();
                            auto locator = rpmc::if_else_c(localityC, makeLogLocator(), rpmc::DefaultLocator{ });
                            auto geometricModelParams = rpmc::GeometricModelParams{
                                .MStar =  castAttr<double>({ args, "args" }, { "star", "M" }),
                                .drMin = castAttr<double>({ args, "args" }, { "zones", "ΔrMin" }),
                                .relativeChangeRate = castAttr<double>({ args, "args" }, { "simulation", "velocity_growth_factor" }),
                                .maxChangeRate = castAttr<double>({ args, "args" }, { "simulation", "velocity_growth_rate" }),
                                .NThreshold = castAttr<double>({ args, "args" }, { "simulation", "N_threshold" }),
                                .StNBodyThreshold = castAttr<double>({ args, "args" }, { "simulation", "St_NBody_threshold" }),
                                .mNBodyThreshold = castAttr<double>({ args, "args" }, { "simulation", "m_NBody_threshold" }),
                                .StDustThreshold = castAttr<double>({ args, "args" }, { "simulation", "St_dust_threshold" }),
                                .mDustThreshold = castAttr<double>({ args, "args" }, { "simulation", "m_dust_threshold" }),
                                .suppress = (effects & Effects::stirring) == Effects::none
                            };
                            auto geometricModelState = rpmc::GeometricModelState{
                                .M = M,
                                .m = m,
                                .N = N,
                                .a = a,
                                .e = e,
                                .sininc = sininc,
                                .St = St,
                                .rho = rho,
                                .vr = vr,
                                .vphi = vphi,
                                .hd = hd
                            };
                            auto stirringModel = rpmc::StirringInteractionModel(geometricModelParams, geometricModelState);
                            auto frictionModel = rpmc::FrictionInteractionModel(geometricModelParams, geometricModelState);
                            return rpmc::RPMCOperator(
                                rpmc::RPMCOperatorParams{
                                    .removalBucketUpdateDelay = castAttr<double>({ args, "args" }, { "simulation", "removal_bucket_update_delay" }),
                                    .rejectionBucketUpdateDelay = castAttr<double>({ args, "args" }, { "simulation", "rejection_bucket_update_delay" })
                                },
                                particleData,
                                RNG(castAttr<unsigned int>({ args, "args" }, { "simulation", "random_seed" }) + 1),
                                classifier,
                                locator,
                                std::move(stirringModel), std::move(frictionModel));
                        },
                        localityV);
                };
                auto stirringOperator = (effects & (Effects::stirring | Effects::friction)) != Effects::none
                    ? makeStirringOperator()
                    : rpmc::PDiscreteOperator(rpmc::NoOpDiscreteOperator{ });
                //auto stirringOperator = rpmc::PDiscreteOperator(rpmc::NoOpDiscreteOperator{ });

                return pybind11ext::make_unique_type<"RPMCSimulation">(
                    rpmc::EventDrivenSimulation(
                        std::tuple{ },
                        std::tuple{ std::move(collisionOperator), std::move(stirringOperator) }));
            }),
        py::keep_alive<1, 3>{ },
        py::arg("args"), py::arg("state"), py::arg("log") = false, py::arg("display") = false);
    using CPyRPMCSimulation = decltype(pyRPMCSimulation)::type;
    pyRPMCSimulation.def("run_to",
        [](CPyRPMCSimulation& self,
           double tEnd,
           py::array_t<double> snapshotTimes, py::object snapshotCallback)
        {
            auto snapshotTimesSpan = py_rpmc::asSpan(py_rpmc::NamedObject{ snapshotTimes, "snapshot_times" });
            auto _1 = py::gil_scoped_release{ };
            self.runTo(tEnd, snapshotTimesSpan,
                [&snapshotCallback]
                (gsl::index i, double t)
                {
                    auto _2 = py::gil_scoped_acquire{ };
                    snapshotCallback(i, t);
                });
        },
        py::arg("t_end"), py::arg("snapshot_times"), py::arg("snapshot_callback"));
    pyRPMCSimulation.def("inspect",
        [](CPyRPMCSimulation const& self,
           py::object dst,
           std::string_view quantity,
           py::object params)
        {
            auto inspectionData = rpmc::InspectionData{ };
            if (py::array_t<double>::check_(dst))
            {
                inspectionData.fdst = py_rpmc::asRaveledSpan<double, py::array::c_style>({ dst, "dst" });
            }
            else if (py::array_t<gsl::index>::check_(dst))
            {
                inspectionData.idst = py_rpmc::asRaveledSpan<gsl::index, py::array::c_style>({ dst, "dst" });
            }
            if (py::array_t<double>::check_(params))
            {
                inspectionData.fparams = py_rpmc::asRaveledSpan<double const, py::array::c_style>({ params, "params" });
            }
            else if (py::array_t<gsl::index>::check_(params))
            {
                inspectionData.iparams = py_rpmc::asRaveledSpan<gsl::index const, py::array::c_style>({ params, "params" });
            }
            bool handled = self.inspect(quantity, inspectionData);
            if (!handled)
            {
                throw std::runtime_error(fmt::format("error in inspect() call: unknown quantity '{}'", quantity));
            }
        },
        py::arg("dst"), py::arg("quantity"), py::arg("params"));
}
