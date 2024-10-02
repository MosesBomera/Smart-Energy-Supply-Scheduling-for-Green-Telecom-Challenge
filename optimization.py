import numpy as np

class EnergyOptimizer:
    """
    A class to optimize energy usage for a telecom base station.

    This class manages the energy flow between solar panels, the power grid,
    a battery, and a diesel generator to meet the energy demands of a telecom base station.

    Parameters
    ----------
    battery_capacity : float
        The capacity of the battery in Amp-hours (Ah).
    rated_voltage : float
        The rated voltage of the battery in Volts (V).
    charge_coefficient : float
        The charging efficiency of the battery (between 0 and 1).
    discharge_coefficient : float
        The discharging efficiency of the battery (between 0 and 1).
    init_soc : float
        The initial state of charge of the battery (between 0 and 1).
    dod : float
        The depth of discharge limit for the battery (between 0 and 1).
    generator_power_kw : float
        The power output of the diesel generator in kilowatts (kW).

    Attributes
    ----------
    soc : float
        The current state of charge of the battery.
    """

    def __init__(self, battery_capacity, rated_voltage, charge_coefficient, 
                 discharge_coefficient, init_soc, dod, generator_power_kw):
        self.battery_capacity = battery_capacity
        self.rated_voltage = rated_voltage
        self.charge_coefficient = charge_coefficient
        self.discharge_coefficient = discharge_coefficient
        self.soc = init_soc
        self.dod = dod
        self.generator_power_kw = generator_power_kw

    def optimize_energy(self, solar_kwh, grid_kwh, load_kwh):
        """
        Optimize the energy usage for a single time step.

        This method determines the optimal energy flow between different sources
        to meet the load demand while minimizing generator usage and maximizing
        battery utilization.

        Parameters
        ----------
        solar_kwh : float
            The solar energy available in kilowatt-hours.
        grid_kwh : float
            The grid energy available in kilowatt-hours.
        load_kwh : float
            The energy demand of the load in kilowatt-hours.

        Returns
        -------
        tuple
            A tuple containing:
            - soc (float): The updated state of charge of the battery.
            - battery_energy_kwh (float): The energy stored in the battery in kilowatt-hours.
            - generator_used (bool): Whether the generator was used or not.
            - generator_energy (float): The energy produced by the generator in kilowatt-hours.
        """
        available_energy = solar_kwh + grid_kwh
        energy_deficit = load_kwh - available_energy
        generator_used = False
        generator_energy = 0
        
        battery_energy_kwh = self.soc * self.battery_capacity * self.rated_voltage / 1000
        max_charge_energy = (1 - self.soc) * self.battery_capacity * self.rated_voltage / 1000

        if energy_deficit > 0:
            if self.soc > self.dod and battery_energy_kwh >= energy_deficit:
                self.update_soc(-energy_deficit)
            else:
                generator_used = True
                generator_energy = min(self.generator_power_kw, energy_deficit)
                remaining_deficit = energy_deficit - generator_energy
                if remaining_deficit > 0 and self.soc > self.dod:
                    battery_contribution = min(battery_energy_kwh, remaining_deficit)
                    self.update_soc(-battery_contribution)
                    remaining_deficit -= battery_contribution
                if remaining_deficit > 0:
                    generator_energy += remaining_deficit
        else:
            charge_energy = min(-energy_deficit, max_charge_energy)
            self.update_soc(charge_energy)

        if generator_used:
            remaining_generator_capacity = self.generator_power_kw - generator_energy
            if remaining_generator_capacity > 0 and self.soc < 1:
                additional_charge = min(remaining_generator_capacity, max_charge_energy)
                self.update_soc(additional_charge)
                generator_energy += additional_charge

        return self.soc, battery_energy_kwh, generator_used, generator_energy

    def update_soc(self, energy_delta_kwh):
        """
        Update the state of charge of the battery.

        Parameters
        ----------
        energy_delta_kwh : float
            The change in energy of the battery in kilowatt-hours.
            Positive for charging, negative for discharging.
        """
        coefficient = self.charge_coefficient if energy_delta_kwh > 0 else self.discharge_coefficient
        delta_soc = (energy_delta_kwh * coefficient) / (self.battery_capacity * self.rated_voltage / 1000)
        self.soc = np.clip(self.soc + delta_soc, self.dod, 1.0)

def run_optimization(solar_kwh, grid_kwh, load_kwh, optimizer):
    """
    Run the energy optimization for a series of time steps.

    Parameters
    ----------
    solar_kwh : array_like
        Array of solar energy available for each time step in kilowatt-hours.
    grid_kwh : array_like
        Array of grid energy available for each time step in kilowatt-hours.
    load_kwh : array_like
        Array of load energy demand for each time step in kilowatt-hours.
    optimizer : EnergyOptimizer
        An instance of the EnergyOptimizer class.

    Returns
    -------
    list of dict
        A list of dictionaries, each containing the results for one time step:
        - 'hour': The hour of the time step.
        - 'soc': The state of charge of the battery.
        - 'battery_energy_kwh': The energy stored in the battery in kilowatt-hours.
        - 'generator_used': Whether the generator was used.
        - 'generator_energy': The energy produced by the generator in kilowatt-hours.
    """
    results = []
    for i in range(len(solar_kwh)):
        soc, battery_energy, generator_used, generator_energy = optimizer.optimize_energy(solar_kwh[i], grid_kwh[i], load_kwh[i])
        results.append({
            'hour': i,
            'soc': soc,
            'battery_energy_kwh': battery_energy,
            'generator_used': generator_used,
            'generator_energy': generator_energy
        })
    return results