
# Given a subset of traits, the functions in this module can infer all implicitly defined traits.


import numpy as np
from typing import Tuple, Optional

import const.cgs as cgs


def compute_dist_traits(
        Δx: Optional[float],
        Δx_2: Optional[float],
        context: str = '') \
            -> Tuple[float, float]:  # (Δx, Δx_2)
    """Given a sufficient subset of distribution traits, infer the values of the others.  
       For example, given Δx_2, infer Δx.

       Keyword arguments:  
       Δx -- distribution width  
       Δx_2 -- distribution half-width  
       context -- context string for error messages
    """
    prefix = context + (': ' if context != '' else '')
    if Δx is None and Δx_2 is None:
        raise RuntimeError(prefix + 'distribution traits are underdetermined, must specify either Δx or Δx_2')
    elif Δx is not None and Δx_2 is not None:
        raise RuntimeError(prefix + 'distribution traits are overdetermined, cannot specify both Δx and Δx_2')
    elif Δx_2 is not None:
        Δx = 2*Δx_2
    else:  # Δx is not None
        Δx_2 = Δx/2
    return Δx, Δx_2


def compute_intrinsic_traits(
        m: Optional[float],
        M: Optional[float],
        ρ: Optional[float],
        N: Optional[int],
        R: Optional[float],
        context: str = '') \
            -> Tuple[float, float, float, int, float]:  # (m, M, ρ, N, R)
    """Given a sufficient subset of particle intrinsic particle traits, infer the values of the others.  
       For example, given (ρ, M, N), infer m and R.

       Keyword arguments:  
       m -- individual particle mass  
       M -- total mass of particles  
       ρ -- particle bulk density  
       N -- total number of particles  
       R -- bulk radius of particles  
       context -- context string for error messages
    """
    prefix = context + (': ' if context != '' else '')

    # M, m, (ρ | R) | M, N, (ρ | R) | m, N, (ρ | R) | M, ρ, R
    if ρ is not None and R is not None:
        if m is not None:
            raise RuntimeError(prefix + 'intrinsic traits are overdetermined, cannot specify all of ρ, R, and m')
        m = 4/3*np.pi*ρ*R**3
        if M is not None and N is not None:
            raise RuntimeError(prefix + 'intrinsic traits are overdetermined, cannot specify all of ρ, R, M, and N')
        elif M is None and N is None:
            raise RuntimeError(prefix + 'intrinsic traits are underdetermined, must specify either M or N if ρ and R are given')
        elif M is not None:
            N = M//m
        else:  # N is not None
            M = N*m
    elif ρ is None and R is None:
        raise RuntimeError(prefix + 'intrinsic traits are underdetermined, at least one of ρ, R must be specified')
    else:
        numMmN = int(M is not None) + int(m is not None) + int(N is not None)
        if numMmN == 3:
            raise RuntimeError(prefix + 'intrinsic traits are overdetermined, cannot specify all of M, m, and N')
        elif numMmN < 2:
            raise RuntimeError(prefix + 'intrinsic traits are underdetermined, must specify two of M, m, and N if ρ or R are given')
        if M is not None and m is not None:
            N = M//m
        elif M is not None and N is not None:
            m = M/N
        else:  # m is not None and N is not None
            M = N*m
        if ρ is not None:
            R = (m/(4/3*np.pi*ρ))**(1/3)
        else:  # R is not None
            ρ = m/(4/3*np.pi*R**3)
    return m, M, ρ, N, R


def compute_kinetic_traits(
        e: Optional[float],
        i: Optional[float],
        sini: Optional[float],
        Δv: Optional[float],
        Δv_vh: Optional[float],
        m: float,
        r: float,
        Mstar: float,
        context: str = '') \
            -> Tuple[float, float, float]:  # (e, i, Δv)
    """Given a sufficient subset of kinetic particle traits, infer the values of the others.  
       For example, given (i, a), infer (e, Δv).

       Keyword arguments:  
       e -- square-averaged eccentricity √<e²>  
       i -- square-averaged inclination angle arcsin √<sin² i>  
       sini -- square-averaged inclination √<sin² i>  
       Δv -- square-averaged thermal velocity √<Δv²> (cm/s)  
       Δv_vh -- square-averaged thermal velocity √<Δv²> in units of Hill velocity  
       m -- particle mass (g)  
       r -- particle orbital radius (cm)  
       Mstar -- mass of central object (g)  
       context -- context string for error messages
    """
    prefix = context + (': ' if context != '' else '')
    if e is None and i is None and sini is None and Δv is None and Δv_vh is None:
        raise RuntimeError(prefix + 'kinetic traits are underdetermined, must specify at least one of e, i, sini, Δv, Δv_vh')
    if i is not None and sini is not None:
        raise RuntimeError(prefix + 'kinetic traits are overdetermined, cannot specify both i and sini')
    if Δv is not None and Δv_vh is not None:
        raise RuntimeError(prefix + 'kinetic traits are overdetermined, cannot specify both Δv and Δv_vh')
    vK = np.sqrt(cgs.GG*cgs.MS/r)  # Kepler velocity of planetesimal (cm/s)
    #  <Δv²>/vK² = <e²> + <i²>  and
    #  <e²> = 2 <i²>  for a truly thermal  Δv  ensemble
    rh = (m/(3*Mstar))**(1/3)
    vh = vK*rh
    if e is None and i is None and sini is None:  # Δv is not None or Δv_vh is not None
        if Δv_vh is not None:
            Δv = Δv_vh*vh
        else:
            Δv_vh = Δv/vh
        sini = np.sqrt(1/3)*Δv/vK
        i = np.arcsin(sini)
        e = np.sqrt(2)*sini
    else:
        if Δv is not None or Δv_vh is not None:
            raise RuntimeError(prefix + 'kinetic traits are overdetermined, cannot specify e or i if Δv or Δv_vh is given')
        if e is None:
            if sini is None:
                sini = np.sin(i)
            else:
                i = np.arcsin(sini)
            e = np.sqrt(2)*sini
        else:
            sini = np.sqrt(1/2)*e
            i = np.arcsin(sini)
        Δv = vK*np.sqrt(e**2 + sini**2)
        Δv_vh = Δv/vh
    return e, i, sini, Δv, Δv_vh


def compute_gas_traits(
        Σ0: Optional[float],
        ρ0: Optional[float],
        T0: Optional[float],
        Tmin: Optional[float],
        cs0: Optional[float],
        Hp0: Optional[float],
        hp0: Optional[float],
        M: Optional[float],
        rlo: Optional[float],
        rhi: Optional[float],
        MStar: float,
        LStar: float,
        φ: float,
        r0: float,
        p: float,
        context: str = ''):

    assert MStar is not None
    assert LStar is not None
    assert φ is not None
    assert r0 is not None
    assert p is not None

    prefix = context + (': ' if context != '' else '')

    μ = 2.3

    Firr = 0.5*np.sin(φ)*LStar/(4*np.pi*r0**2)  # irradiating flux  (factor 1/2: two-layer model!)
    TirrQd = Firr/cgs.ss  # quadrupled irradiative mid-plane temperature  Tirr⁴  (K⁴)

    ΩK0 = np.sqrt(cgs.GG*MStar/(r0**3))  # Kepler rotation frequency  (1/s)

    if cs0 is not None or Hp0 is not None or hp0 is not None:
        if Hp0 is not None and hp0 is not None:
            raise RuntimeError(prefix + 'cannot specify both Hp0 and hp0')
        if cs0 is not None and (Hp0 is not None or hp0 is not None):
            raise RuntimeError(prefix + 'cannot specify cₛ0 if Hp0 or hp0 was specified')
        if T0 is not None:
            raise RuntimeError(prefix + 'cannot specify T0 if cₛ0, Hp0, or hp0 was specified')
        if Tmin is not None:
            raise RuntimeError(prefix + 'cannot specify Tmin if cₛ0, Hp0, or hp0 was specified')
        if cs0 is None:
            if Hp0 is None:
                Hp0 = hp0*r0  # pressure scale height  (cm)
            else:
                hp0 = Hp0/r0  # dimensionless pressure scale height
            cs0 = ΩK0*Hp0  # sound speed  (cm/s)
        else:
            Hp0 = cs0/ΩK0  # pressure scale height  (cm)
            hp0 = Hp0/r0  # dimensionless pressure scale height
        T0 = cs0**2/cgs.kk*(μ*cgs.mp)  # effective mid-plane temperature  (K)
        if T0**4 < TirrQd:
            print("T0 = {} K\nTirrQd = {} K".format(T0, TirrQd**(1/4)))
            raise RuntimeError(prefix + 'with the specified value of cₛ0, the temperature T0 is lower than the irradiative mid-plane temperature')
        Tmin = (T0**4 - TirrQd)**(1/4)  # background temperature  (K)
    else:
        if cs0 is None and Tmin is None and T0 is None:
            raise RuntimeError(prefix + 'gas traits are underdetermined, must specify at least one of cs0, Hp0, hp0, T0, Tmin')
        if Tmin is not None and T0 is not None:
            raise RuntimeError(prefix + 'cannot specify both T0 and Tmin')
        if T0 is not None:
            if T0**4 < TirrQd:
                print("T0 = {} K\nTirrQd = {} K".format(T0, TirrQd**(1/4)))
                raise RuntimeError(prefix + 'specified value of T0 is lower than the irradiative mid-plane temperature')
            Tmin = (T0**4 - TirrQd)**(1/4)  # background temperature  (K)
        else:  # Tmin is not None
            T0 = (Tmin**4 + TirrQd)**(1/4)  # effective mid-plane temperature  (K)
        cs0 = np.sqrt(cgs.kk*T0/(μ*cgs.mp))
        Hp0 = cs0/ΩK0  # pressure scale height  (cm)
        hp0 = Hp0/r0  # dimensionless pressure scale height

    if M is not None:
        if rlo is None or rhi is None:
            raise RuntimeError(prefix + 'rₗₒ and rₕᵢ must be given if M is specified')
        if Σ0 is not None or ρ0 is not None:
            raise RuntimeError(prefix + 'gas traits are overdetermined, cannot specify Σ0 or ρ0 if M is given')
        Σ0 = M*(p + 2)/(2*np.pi)*r0**p/(rhi**(p + 2) - rlo**(p + 2))
    elif Σ0 is None:
        if ρ0 is None:
            raise RuntimeError(prefix + 'must specify either Σ0, ρ0, or M and rmin and rmax')
        Σ0 = ρ0*(np.sqrt(2*np.pi)*Hp0)
    elif ρ0 is not None:
        raise RuntimeError(prefix + 'gas traits are overdetermined, cannot specify both Σ0 and ρ0')
    if ρ0 is None:
        ρ0 = Σ0/(np.sqrt(2*np.pi)*Hp0)
    return Σ0, ρ0, T0, Tmin, cs0, Hp0, hp0


def compute_dust_trapping_planet_traits(
        rPlanet: Optional[float],
        rDustTrap: Optional[float],
        gas_profile_base_args,
        mPlanet: float,
        context: str = ''):

    import rpmc

    assert mPlanet is not None

    prefix = context + (': ' if context != '' else '')
    
    if rPlanet is None and rDustTrap is None:
        raise RuntimeError(prefix + 'planet traits are underdetermined: either rPlanet or rDustTrap must be specified')
    if rPlanet is not None and rDustTrap is not None:
        raise RuntimeError(prefix + 'planet traits are overdetermined: cannot specify both rPlanet and rDustTrap')

    St = 1.

    if rPlanet is not None:
        rDustTrap = rpmc.compute_dust_trap_position_for_power_law_with_planetary_gap_gas_profile(
            gas_profile_base_args=gas_profile_base_args, m_planet=mPlanet, r_planet=rPlanet, St=St)
    else:  # rDustTrap is not None
        rPlanet = rpmc.compute_planet_position_for_power_law_with_planetary_gap_gas_profile(
            gas_profile_base_args=gas_profile_base_args, m_planet=mPlanet, r_dust_trap=rDustTrap, St=St)
        if np.isnan(rPlanet):
            raise RuntimeError(prefix + 'cannot determine position of planet of given mass for dust trap at given position')

    return rPlanet, rDustTrap


def compute_planet_traits(
        ρ: Optional[float],
        R: Optional[float],
        e: Optional[float],
        i: Optional[float],
        sini: Optional[float],
        m: float,
        context: str = '') \
            -> Tuple[float, float, float, float]:  # (ρ, R, e, i)

    assert m is not None

    prefix = context + (': ' if context != '' else '')

    if ρ is not None and R is not None:
        raise RuntimeError(prefix + 'planet traits are overdetermined, cannot specify ρ and R')
    elif ρ is None and R is None:
        raise RuntimeError(prefix + 'planet traits are underdetermined, at least one of ρ and R must be specified')
    else:
        if ρ is not None:
            R = (m/(4/3*np.pi*ρ))**(1/3)
        else:  # R is not None
            ρ = m/(4/3*np.pi*R**3)

    if e is None:
        e = 0.
    if i is None:
        if sini is not None:
            i = np.arcsin(sini)
        else:
            sini = 0.
            i = 0.
    else:
        if sini is not None:
            raise RuntimeError(prefix + 'planet traits are overdetermined, cannot specify i and sini')
        sini = np.sin(i)

    return ρ, R, e, i, sini
