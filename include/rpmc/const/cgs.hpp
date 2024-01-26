
// Constants in the CGS unit system.


#ifndef INCLUDED_RPMC_CONST_CGS_HPP_
#define INCLUDED_RPMC_CONST_CGS_HPP_


namespace rpmc::cgs {


    // Natural constants in CGS
constexpr double GG   = 6.67408e-08;     // Gravitational constant  (cm³/g/s²)
constexpr double mp   = 1.6726e-24;      // Mass of proton          (g)
constexpr double me   = 9.1095e-28;      // Mass of electron        (g)
constexpr double kk   = 1.3807e-16;      // Bolzmann's constant     (erg/K)
constexpr double hh   = 6.6262e-27;      // Planck's constant       (erg s)
constexpr double ee   = 4.8032e-10;      // Unit charge             
constexpr double cc   = 2.9979245800000e+10;  // Light speed        (cm/s)
constexpr double st   = 6.6524e-25;      // Thomson cross-section   (cm²)
constexpr double ss   = 5.6703e-5;       // Stefan-Boltzmann const  (erg/cm²/K⁴/s)
constexpr double aa   = 7.5657e-15;      // 4 ss / cc               (erg/cm³/K⁴)

    // Gas constants
constexpr double muh2 = 2.3000e+0;       // Mean molec weight H2+He+Metals
constexpr double mole = 6.02214129e+23;  // 1 Mole (Avogadro's constant) = 1/mu

    // Alternative units
constexpr double ev   = 1.6022e-12;      // Electronvolt            (erg)
constexpr double kev  = 1.6022e-9;       // Kilo electronvolt       (erg)
constexpr double micr = 1.e-4;           // Micron                  (cm)
constexpr double km   = 1.e+5;           // Kilometer               (cm)
constexpr double angs = 1.e-8;           // Angstroem               (cm)

    // Astronomy constants
constexpr double LS  = 3.8525e+33;       // Solar luminosity        (erg/s)
constexpr double RS  = 6.96e+10;         // Solar radius            (cm)
constexpr double MS  = 1.98892e+33;      // Solar mass              (g)
constexpr double TS  = 5.78e+3;          // Solar temperature       (K)
constexpr double AU  = 1.49598e+13;      // Astronomical Unit       (cm)
constexpr double au  = 1.49598e+13;      // Astronomical Unit       (cm)
constexpr double pc  = 3.08572e+18;      // Parsec                  (cm)
constexpr double Mea = 5.9736e+27;       // Mass of Earth           (g)
constexpr double Rea = 6.375e+08;        // Equatorial radius Earth (cm)
constexpr double Mmo = 7.347e+25;        // Mass of moon            (g)
constexpr double Rmo = 1.738e+08;        // Radius of moon          (cm)
constexpr double dmo = 3.844e+10;        // Distance earth-moon (center-to-center)  (cm)
constexpr double Mju = 1.899e+30;        // Mass of Jupiter         (g)
constexpr double Rju = 7.1492e+9;        // Equat. radius Jupiter   (rm)
constexpr double dju = 7.78412e+13;      // Distance Jupiter-Sun    (cm)
constexpr double Jy  = 1e-23;            // Jansky                  (erg/s/cm²/Hz)

    // Time units
constexpr double year = 31557600.e+0;    // Year                    (s)
constexpr double hour = 3.6000e+3;       // Hour                    (s)
constexpr double day  = 8.64e+4;         // Day                     (s)

    // Miscellaneous
constexpr double bar  = 1.e+6;           // One bar = 10⁶ barye     (Ba)


} // namespace rpmc::cgs


#endif // INCLUDED_RPMC_CONST_CGS_HPP_
