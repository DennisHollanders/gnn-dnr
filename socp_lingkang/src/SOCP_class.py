import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np
import logging

from pyomo.environ import (
    Reals,
    NonNegativeReals,
    Var,
    Param,
    Constraint,
    ConcreteModel,
    Objective,
    ConstraintList,
    minimize,
    SolverFactory,
    Set,
)

from pyomo.util.infeasible import log_infeasible_constraints
from pyomo.environ import value


class SOCP_class:
    """
    This class prepare the Second Order Conic Programming optimization model, in order to perform the
     optimal power flow,
    it is base as input a pandapower network, which is has been solved in pandapower,
    the network data is properly processed,
    in order to pass it though the Pyomo optimization model.
    Once solved the optimization model, the results is compared with the optimal power flow data from pandapower.

    Requires the following conditions of the network:
    - no overlapping nodes
    - clear flow of the network ( direction)
    - has to be radial one (since I did not include the Kirchoff condition of the closed loops)
    """

    def __init__(
        self,
        net,
        net_name: str,
        mode: str,
        df_load: pd.DataFrame,
        df_gen: pd.DataFrame,
        df_sgen: pd.DataFrame,
    ):
        """
        Initialize the SOCP_class with the network data and name.

        Parameters:
        net (pandapowerNet): The pandapower network data.
        net_name (str): The name of the network.

        Attributes:
        total_active_power_loss (float): Total active power loss in the network.
        """
        self.net = net
        self.net_name = net_name
        active_power_loss = self.net.res_line.pl_mw
        # Total active power loss in the network
        self.total_active_power_loss = sum(
            active_power_loss
        )  # used only for the static simulation
        self.line_df = None
        self.bus_df = None
        self.bus_dict = None
        self.voltages_bus = None
        self.mode = mode
        self.load = df_load
        self.gen = df_gen
        self.sgen = df_sgen

    def initialize(self):
        """
        Process the pandapower network data for later use, including extracting bus information,
        line parameters, and handling transformer data to avoid potential issues in Second Order
        Cone Programming (SOCP).

        The following steps are performed:
        1. Extract bus data and create a dictionary mapping bus indices to bus names.
        2. Exclude high-voltage (hv) side of transformers from the bus data as it is not considered
           in SOCP and might cause problems.
        3. Extract and process voltage levels for each bus.
        4. Extract line data including resistance, reactance, and susceptance per line, and calculate
           their total values based on the line length.
        5. Store processed data as class attributes for further use.
        """
        #  Preparing the pandapower net topology data
        bus_df = self.net.bus
        # getting the dict of the names of the buses associated to the pandapower indexing
        # (Pandapower naming could be messy)
        bus_dict = {
            index: name for index, name in zip(self.net.bus.index, self.net.bus.name)
        }

        # excluding the hv end of the trafo
        # as is not considered in SOCP and could cause problems

        for index, row in self.net.trafo.iterrows():
            hv_bus = bus_dict[row["hv_bus"]]
            condition = bus_df.name == hv_bus
            index = bus_df[condition].index[0]
            bus_df.drop(index, inplace=True)

        # voltages
        voltages_bus = bus_df[["name", "vn_kv"]].set_index("name")

        # % line data extraction, resistance, reactance and susceptance
        line_df = self.net.line
        line_df["r_ohm"] = line_df.apply(
            lambda row: row["r_ohm_per_km"] * row["length_km"], axis=1
        )
        line_df["x_ohm"] = line_df.apply(
            lambda row: row["x_ohm_per_km"] * row["length_km"], axis=1
        )
        frequency = 50  # HZ

        line_df["B_Siemens_shunt"] = line_df.apply(
            lambda row: row["c_nf_per_km"]
            * row["length_km"]
            * 2
            * np.pi
            * frequency
            / 10**9,
            axis=1,  # using SI (F)
        )

        def susceptance_cal(R, X):
            if R == 0:
                return -1 / X
            else:
                return -X / (R**2 + X**2)

        line_df["B_transmission_s"] = line_df.apply(
            lambda row: susceptance_cal(row["r_ohm"], row["x_ohm"]), axis=1
        )

        bus_susceptance = {bus: 0 for bus in self.net.bus.index}

        # Process each line
        for idx, row in self.net.line.iterrows():
            B = row["c_nf_per_km"] * row["length_km"] * 2 * np.pi * frequency * 1e-9

            # Add the susceptance to both connected buses
            bus_susceptance[row["from_bus"]] += B
            bus_susceptance[row["to_bus"]] += B

        B_df = pd.DataFrame()

        B_df["bus_n"] = bus_susceptance.keys()
        B_df["bus_B_s"] = bus_susceptance.values()
        B_df["bus_name"] = [bus_dict[i] for i in B_df["bus_n"]]

        # passing to the class
        self.line_df = line_df
        self.bus_df = bus_df
        self.bus_dict = bus_dict
        self.voltages_bus = voltages_bus
        self.B_df = B_df

    def matrix_creation_from_pp(self, var, model, df):
        """
        Create a matrix based on input data (power values) for static or dynamic (time-series) scenarios.

        This method generates a matrix that represents power values (active and reactive power)
        for a set of buses, either for a single time period ('static' mode) or multiple time periods
        ('dynamic' mode). In 'static' mode, both active (p_mw) and reactive (q_mvar) power are included,
        while in 'dynamic' mode, the method currently only supports active power.

        Parameters:
        -----------
        var : pandas.DataFrame
            Input data containing active and reactive power values ('p_mw', 'q_mvar') and bus information for static scenarios.
            Each row corresponds to a bus, and the columns contain the power data.

        model : Pyomo.ConcreteModel
            A Pyomo model instance containing the optimization model information.
            The 'times' attribute from the model provides the number of time steps for dynamic scenarios.

        df : pandas.DataFrame
            Input data for dynamic scenarios (time-series).
            The rows represent time steps, and the columns represent the bus indices for which the power values are given.

        Returns:
        --------
        pandas.DataFrame
            A matrix with power values (p_mw, q_mvar) for static mode, or a time-series matrix with active power (p_mw) for dynamic mode.
            The matrix includes the following columns:
            - 'time': The time step (for dynamic mode) or repeated 0 for static mode.
            - 'Bus': The bus name or identifier.
            - 'p_mw': Active power at each bus.
            - 'q_mvar': Reactive power (only in static mode).

        Notes:
        ------
        - For static mode, the method processes input data (var) to create a matrix for a single time step,
          with both active and reactive power values.
        - For dynamic mode, the method handles time-series input data (df), repeating buses across time steps,
          and currently only supports active power ('p_mw').
        - If the input data is missing reactive power ('q_mvar') in static mode, it defaults to zero.
        - In both modes, missing values in the matrix are filled with zeros.
        - The 'lv bus' from the transformer is excluded in the bus information for optimality reasons.
        - Reactive power is not yet included for dynamic scenarios.
        """

        if self.mode == "static":
            matrix = pd.DataFrame(
                index=self.bus_df.name.values, columns=["p_mw", "q_mvar"]
            )
            for index, row in var.iterrows():
                bus_name = self.bus_dict[row["bus"]]
                matrix.loc[bus_name, "p_mw"] = row["p_mw"]
                try:
                    matrix.loc[bus_name, "q_mvar"] = row["q_mvar"]
                except:
                    matrix.loc[bus_name, "q_mvar"] = 0

            len_time = len(model.times)

            matrix.fillna(0, inplace=True)
            matrix.index.name = "Bus"
            time_column = np.tile(np.arange(0, len_time), len(matrix))
            matrix_time = matrix.loc[matrix.index.repeat(len_time)].reset_index(
                drop=False
            )
            matrix_time["time"] = time_column
        else:
            # dynamic/time series
            # todo: reactive power is not included at the current stage

            # getting the name based on the index of the bus index
            bus_dict_reversed = dict(zip(self.net.bus.index, self.net.bus.name))
            # creating a list of the time steps based on the time index and number of buses present
            time_column = list(df.index) * len(self.net.bus)
            # creting buses, repeated by the times of the time indexes
            bus = [
                item for item in list(self.net.bus.name) for _ in range(len(df.index))
            ]

            # initialization of the matrix to be created
            matrix_time = pd.DataFrame()
            matrix_time["time"] = time_column
            matrix_time["Bus"] = bus
            matrix_time["p_mw"] = [0] * len(matrix_time)

            # substitue the data with the data import from pandapower (reactive power is not included at the current stage)
            for row in df.index:
                for column in df.columns:
                    bus_name = bus_dict_reversed[self.net.load.loc[column].bus]
                    condition = (matrix_time["time"] == row) & (
                        matrix_time["Bus"] == bus_name
                    )
                    matrix_time.loc[condition, "p_mw"] = df.loc[row, column]
            # avoiding error
            matrix_time.fillna(0, inplace=True)
        return matrix_time

    def Pyomo_model_creation_from_pp(self, n_ts):
        """
        Creates a Pyomo model for optimal power flow using Second-Order Cone Programming (SOCP) based on
        the grid data from a Pandapower network.

        This method initializes a Pyomo ConcreteModel and sets up various components of the model, including:

        - **Sets**: Defines the sets for lines, buses, and time steps in the power grid.
        - **Parameters**: Introduces parameters such as voltage limits, line resistances, reactances, and susceptances.
        - **Variables**: Declares variables for power flows, losses, and squared voltage magnitudes.
        - **Constraints**: Establishes power balance constraints, voltage limits, and SOCP-based conditions for power flows.
        - **Objective**: Minimizes power losses across the grid by optimizing the model's variables.

        The units used are in miliohm, kW, and V to maintain coherence with the underlying methodology.

        **Parameters**:
        - `n_ts`: A list of time steps used in the model. This is passed when the mode is dynamic, otherwise, a static model is created.

        **Main components**:
        - **Voltage limits**: Upper and lower bounds on the bus voltages, scaled by 5% deviation.
        - **Line characteristics**: Resistances, reactances (in milliohms), and susceptances (in kilosiemens).
        - **Power flows**: Variables for real and reactive power flows in and out of each bus.
        - **Losses**: Variables for active and reactive power losses across lines.
        - **Primary and secondary matrices**: Matrices representing the power grid topology (e.g., connectivity between buses and lines).
        - **Power balance**: Ensures that the net power flow in each bus equals the difference between power import and export.
        - **Voltage balance**: Represents the voltage drops along the lines due to power losses.

        **Objective**:
        - Minimize total active power losses (P_tilde_loss) across all lines and time steps.

        **Returns**:
        - A Pyomo ConcreteModel instance encapsulating the power flow problem for optimization using Gurobi or other solvers.

        Notes:
        - The model excludes the low-voltage buses of transformers (trafo) from certain power flow constraints to ensure correctness.
        - It introduces auxiliary constraints for modeling power exports and imports at different buses.
        - Load data is incorporated directly from Pandapower network data, including special handling for static generators (sgen) and regular generators (gen).

        Raises:
        - If reactive power is not considered, an appropriate message is displayed indicating that it will be added later.

        Example:
        - This method is typically used in power grid optimization scenarios to calculate optimal power flow, minimize losses, and ensure grid stability.
        """

        #
        #  Pyomo model

        # initialization
        # Create a logger to capture infeasibility details
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("pyomo.util.infeasible")

        # Pyomo model
        """"
        The units adopted are in miliohm, kW, and V"""

        model = ConcreteModel()

        # Define constants for conversion factors
        KV_TO_V = 1000  # Kilovolts to Volts
        OHM_TO_MOHM = 1000  # Ohms to milliohms
        MW_TO_KW = 1000  # MW to kW
        S_TO_KS = 1000  # Siemens to kilosiemens
        VOLTAGE_LIMIT_MULTIPLIER_OVER = 1.05  # Upper voltage limit multiplier
        VOLTAGE_LIMIT_MULTIPLIER_UNDER = 0.95  # Lower voltage limit multiplier

        # sets
        model.lines = Set(initialize=list(self.line_df["name"]), ordered=True)

        model.buses = Set(initialize=list(self.bus_df["name"]), ordered=True)
        if self.mode == "static":
            model.times = Set(initialize=[i for i in range(0, 1, 1)], ordered=True)
        else:
            model.times = Set(initialize=[i for i in n_ts], ordered=True)

        # grid topology params
        V_ref_sqr = (
            self.voltages_bus.vn_kv.unique()[0] * KV_TO_V
        ) ** 2  # Reference voltage in Volts squared

        # Voltage limits
        model.V_overline = Param(
            model.buses,
            within=NonNegativeReals,
            initialize=self.voltages_bus
            * VOLTAGE_LIMIT_MULTIPLIER_OVER
            * KV_TO_V,  # Upper voltage limit in Volts
        )
        model.V_underline = Param(
            model.buses,
            within=NonNegativeReals,
            initialize=self.voltages_bus
            * VOLTAGE_LIMIT_MULTIPLIER_UNDER
            * KV_TO_V,  # Lower voltage limit in Volts
        )

        # Line parameters: reactance and resistance in milliohms
        model.xl_mOhm = Param(
            model.lines,
            within=NonNegativeReals,
            initialize=self.line_df.set_index("name")["x_ohm"]
            * OHM_TO_MOHM,  # Convert reactance to milliohms
        )
        model.rl_mOhm = Param(
            model.lines,
            within=NonNegativeReals,
            initialize=self.line_df.set_index("name")["r_ohm"]
            * OHM_TO_MOHM,  # Convert resistance to milliohms
        )

        # Susceptance (capacitive part of cables) in kilosiemens
        model.B_kSiemens = Param(
            model.buses,
            within=Reals,
            initialize=self.B_df.set_index("bus_name")["bus_B_s"]
            / S_TO_KS,  # Convert Siemens to kilosiemens
        )

        # primary_matrix for modeling (see paper for better understanding: https://ieeexplore.ieee.org/document/6826586)
        M_f = pd.DataFrame(
            index=self.bus_df.name.values, columns=list(self.net.line.name)
        )
        print("M_f", M_f.shape)

        for index, row in self.net.line.iterrows():
            M_f.at[self.bus_dict[row["from_bus"]], row["name"]] = -1
            M_f.at[self.bus_dict[row["to_bus"]], row["name"]] = +1
        M_f.fillna(0, inplace=True)
        
        M_l = pd.DataFrame(
            index=self.bus_df.name.values, columns=list(self.net.line.name)
        )
        for index, row in self.net.line.iterrows():
            M_l.loc[self.bus_dict[row["from_bus"]]].loc[row["name"]] = +1
        M_l.fillna(0, inplace=True)

        M_w = M_f.T  # primary_matrix for the square of voltages

        self.M_f = M_f
        self.M_l = M_l
        self.M_w = M_w
        print("M_f", M_f.shape,)
        print("M_l", M_l.shape)
        print("M_w", M_w.shape)
        # setting as vars
        model.p_export = Var(
            model.buses,
            model.times,
            within=NonNegativeReals,
            initialize=0,
        )
        model.p_import = Var(
            model.buses,
            model.times,
            within=NonNegativeReals,
            initialize=0,
        )
        # net power just as support variable
        model.net_p = Var(
            model.buses,
            model.times,
            within=Reals,
            initialize=0,
        )

        model.q_export = Var(
            model.buses,
            model.times,
            within=NonNegativeReals,
            initialize=0,
        )
        model.q_import = Var(
            model.buses,
            model.times,
            within=NonNegativeReals,
            initialize=0,
        )

        model.net_q = Var(
            model.buses,
            model.times,
            within=Reals,
            initialize=0,
        )

        # model variables (squared voltage)
        model.V_m_sqr = Var(
            model.buses,
            model.times,
            within=NonNegativeReals,
            initialize=1,
        )
        # In SOCP_class.Pyomo_model_creation_from_pp, before defining lv_buses_trafo
        if hasattr(self.net, "trafo") and not self.net.trafo.empty:
            lv_buses_trafo = [self.bus_dict[i] for i in self.net.trafo["lv_bus"].values][0]
        else:
            print("Warning: Transformer table is empty; transformer exclusion constraints will be skipped.")
            lv_buses_trafo = None

        model.P = Var(model.buses, model.times, initialize=0)
        model.Q = Var(model.buses, model.times, initialize=0)

        model.exluding_trafo_ = ConstraintList()
        if lv_buses_trafo is not None:
            for time in model.times:
                model.exluding_trafo_.add(model.P[lv_buses_trafo, time] == 0)
                model.exluding_trafo_.add(model.Q[lv_buses_trafo, time] == 0)

        # auxiliary dict for automatic detection of the receiving bus for each line for active power
        P_receiving_bus_dict = {}
        for line in model.lines:
            for t in model.times:
                bus_name = self.bus_dict[
                    self.net.line[self.net.line["name"] == line]["to_bus"].iloc[0]
                ]
                P_receiving_bus_dict[line, t] = model.P[bus_name, t]

        # auxiliary dict for automatic detection of the receiving bus for each line for reactive power
        Q_receiving_bus_dict = {}
        for line in model.lines:
            for t in model.times:
                bus_name = self.bus_dict[
                    self.net.line[self.net.line["name"] == line]["to_bus"].iloc[0]
                ]
                Q_receiving_bus_dict[line, t] = model.Q[bus_name, t]

        # line variables
        model.P_loss = Var(model.lines, model.times, initialize=0)
        model.Q_loss = Var(model.lines, model.times, initialize=0)
        model.P_tilde_loss = Var(model.lines, model.times, initialize=0)

        # constraints
        def adding_data_as_constraint(
            Cons_list,
            var_string,
            model_var,
            primary_matrix,
            lv_buses_trafo,
            all_present_houses,
            battery_houses_names,
            constraint_limit=10,  # Default constraint for battery houses
        ):
            """
            Adds constraints to the model for buses' active or reactive power based on the given primary matrix.

            :param Cons_list: Pyomo constraints list
            :param var_string: Name of the variable (e.g., 'p_mw' for active power or 'q_mvar' for reactive power)
            :param model_var: Pyomo model variable (e.g., p_import or q_import)
            :param primary_matrix: DataFrame representing the matrix for the power variable
            :param lv_buses_trafo: List of low voltage transformer buses to exclude from constraints
            :param all_present_houses: List of houses where loads are present
            :param battery_houses_names: List of battery house names
            :param constraint_limit: Maximum allowed value for power in battery houses (default is 10 kW)
            """

            for _, row in primary_matrix.iterrows():
                bus = row["Bus"]
                time = row["time"]

                # Exclude transformer buses from constraints
                if bus in lv_buses_trafo:
                    continue

                value = abs(row[var_string] * MW_TO_KW)  # Convert from MW to kW

                if bus in all_present_houses:
                    if bus in battery_houses_names:
                        # todo: here is for these buses to have possibility to change the active power
                        # Cons_list.add(
                        #     model_var[bus, time] <= constraint_limit
                        # )  # Limit to 10 kW for battery houses

                        # todo: here is to activate comparison in pandapower
                        Cons_list.add(model_var[bus, time] == value)
                    else:
                        Cons_list.add(
                            model_var[bus, time] == value
                        )  # Set value for normal houses
                else:
                    Cons_list.add(model_var[bus, time] == 0)

        # load primary_matrix manipulation from the pandapower model
        load_matrix = self.matrix_creation_from_pp(self.net.load, model, df=self.load)
        # gen primary_matrix from pandapower model (voltage regulated generator)
        if self.gen is None:  # if not present
            gen_matrix = load_matrix.copy()
            gen_matrix["p_mw"] = [0] * len(gen_matrix)
        else:
            gen_matrix = self.matrix_creation_from_pp(self.net.gen, model, df=self.gen)
        # sgen primary_matrix from pandapower model (static gen)
        if self.sgen is None:
            sgen_matrix = load_matrix.copy()
            sgen_matrix["p_mw"] = [0] * len(sgen_matrix)
        else:
            sgen_matrix = self.matrix_creation_from_pp(
                self.net.sgen, model, df=self.sgen
            )
        # overall gen_matrix
        all_gen_matrix = gen_matrix + sgen_matrix
        all_gen_matrix["time"] = gen_matrix["time"]
        all_gen_matrix["Bus"] = gen_matrix["Bus"]

        # processing load p
        lv_buses_trafo = [
            self.bus_dict[i] for i in self.net.trafo["lv_bus"].values
        ]  # necessary to exclude the buses of the trafo from constraint

        # Fetch house connections and battery house names
        # Connections = pd.read_pickle("data/Connections.pkl")
        # all_present_houses = list(Connections.Name)
        # battery_houses_names = [
        #     "871694840031194243",
        #     "871694840006109302",
        #     "871694840006302284",
        #     "871694840032895132",
        #     "871694840000077683",
        #     "871694840000077690",
        #     "871694840006108411",
        #     "871694840000018617",
        #     "871694840006352043",
        # ]

        # # Add constraints for active power (p_mw)
        # model.p_buses_cons = ConstraintList()
        # adding_data_as_constraint(
        #     Cons_list=model.p_buses_cons,
        #     var_string="p_mw",
        #     model_var=model.p_import,
        #     primary_matrix=load_matrix,
        #     lv_buses_trafo=lv_buses_trafo,
        #     all_present_houses=all_present_houses,
        #     battery_houses_names=battery_houses_names,
        # )

        # # Add constraints for generator power (p_mw)
        # model.p_buses_cons_2 = ConstraintList()
        # adding_data_as_constraint(
        #     Cons_list=model.p_buses_cons_2,
        #     var_string="p_mw",
        #     model_var=model.p_export,
        #     primary_matrix=all_gen_matrix,
        #     lv_buses_trafo=lv_buses_trafo,
        #     all_present_houses=all_present_houses,
        #     battery_houses_names=battery_houses_names,
        # )

        # # Processing load q (reactive power)
        # try:
        #     model.q_buses_cons = ConstraintList()
        #     adding_data_as_constraint(
        #         Cons_list=model.q_buses_cons,
        #         var_string="q_mvar",
        #         model_var=model.q_import,
        #         primary_matrix=load_matrix,
        #         lv_buses_trafo=lv_buses_trafo,
        #         all_present_houses=all_present_houses,
        #         battery_houses_names=battery_houses_names,
        #     )

        #     model.q_buses_cons_2 = ConstraintList()
        #     adding_data_as_constraint(
        #         Cons_list=model.q_buses_cons_2,
        #         var_string="q_mvar",
        #         model_var=model.q_export,
        #         primary_matrix=all_gen_matrix,
        #         lv_buses_trafo=lv_buses_trafo,
        #         all_present_houses=all_present_houses,
        #         battery_houses_names=battery_houses_names,
        #     )
        # except KeyError as e:
        #     print(f"Reactive power constraint skipped due to missing data: {str(e)}")

        def Var_introduction(model, l, t):
            return 2 * model.rl_mOhm[l] * model.P_tilde_loss[l, t] == model.P_loss[l, t]

        model.Var_introduction_constraint = Constraint(
            model.lines, model.times, rule=Var_introduction
        )

        def net_p(model, b, t):
            return model.net_p[b, t] == model.p_export[b, t] - model.p_import[b, t]

        model.net_p_cons = Constraint(model.buses, model.times, rule=net_p)

        def net_q(model, b, t):
            return model.net_q[b, t] == model.q_export[b, t] - model.q_import[b, t]

        model.net_q_cons = Constraint(model.buses, model.times, rule=net_q)

        def variable_explicit(model, l, t):
            # receiving_bus name
            r = M_f[M_f[l] == +1].index[0]  # receiving
            return (
                2 * model.P_tilde_loss[l, t] * model.V_m_sqr[r, t] * V_ref_sqr
                >= model.P[r, t] ** 2 + model.Q[r, t] ** 2
            )

        model.variable_explicit_cons = Constraint(
            model.lines, model.times, rule=variable_explicit
        )

        def bus_active_power(model, b, t):
            return model.p_export[b, t] - model.p_import[b, t] == sum(
                M_f.loc[b, line] * P_receiving_bus_dict[line, t]
                + M_l.loc[b, line] * model.P_loss[line, t]
                for line in model.lines
            )

        model.bus_active_power_cons = Constraint(
            model.buses, model.times, rule=bus_active_power
        )

        def bus_reactive_power(model, b, t):
            return (
                model.q_export[b, t] - model.q_import[b, t]
                == sum(
                    M_f.loc[b, line] * Q_receiving_bus_dict[line, t]
                    + M_l.loc[b, line] * model.Q_loss[line, t]
                    for line in model.lines
                )
                + model.B_kSiemens[b]
                * model.V_m_sqr[b, t]
                * V_ref_sqr  # to be determined the sign
            )

        model.bus_reactive_power_cons = Constraint(
            model.buses, model.times, rule=bus_reactive_power
        )

        def voltage_loss_over_lines(model, l, t):
            # receiving_bus name
            r = M_f[M_f[l] == 1].index[0]  # receiving +1
            return (
                2 * model.rl_mOhm[l] * model.P[r, t]
                + 2 * model.xl_mOhm[l] * model.Q[r, t]
                + model.rl_mOhm[l] * model.P_loss[l, t]
                + model.xl_mOhm[l] * model.Q_loss[l, t]
                - sum(
                    M_w.loc[l, bus] * model.V_m_sqr[bus, t] * V_ref_sqr
                    for bus in model.buses  # matrix operation
                )  # changed sign compared to the
                == 0
            )

        model.bus_voltage_loss_cons = Constraint(
            model.lines, model.times, rule=voltage_loss_over_lines
        )

        def voltage_limitation_upper(model, b, t):
            return model.V_m_sqr[b, t] * V_ref_sqr <= model.V_overline[b] ** 2

        model.voltage_limitation_cons_upper = Constraint(
            model.buses, model.times, rule=voltage_limitation_upper
        )

        def voltage_limitation_lower(model, b, t):
            return model.V_underline[b] ** 2 <= model.V_m_sqr[b, t] * V_ref_sqr

        model.voltage_limitation_cons_lower = Constraint(
            model.buses, model.times, rule=voltage_limitation_lower
        )

        def loss_calibration(model, l, t):
            return (
                model.xl_mOhm[l] * model.P_loss[l, t]  # exhcanged here
                == model.rl_mOhm[l] * model.Q_loss[l, t]
            )

        model.loss_calibration_cons = Constraint(
            model.lines, model.times, rule=loss_calibration
        )

        def objective(model):
            return sum(
                model.P_tilde_loss[l, t] for l in model.lines for t in model.times
            )

        model.objective = Objective(rule=objective, sense=minimize)

        self.model = model

    def print_problematic_constraints(self):
        for constr in self.model.component_objects(Constraint, active=True):
            for index in constr:
                if not constr[index].active:
                    continue
                if (
                    constr[index].body() != constr[index].lower
                    or constr[index].body() != constr[index].upper
                ):
                    print(
                        f"Constraint {constr.name}[{index}] is problematic: {constr[index].body()} not in [{constr[index].lower}, {constr[index].upper}]"
                    )

    def solving_SOCP_model(self):

        with open("model.txt", "w") as f:
            # Use the file object as the output stream for pprint()
            self.model.pprint(ostream=f)

        # required for debugging
        self.model.write("model.lp", io_options={"symbolic_solver_labels": True})

        # gurobipy part
        import gurobipy as gp

        gurobi_model = gp.read("model.lp")
        # Allow p_import and p_export to vary within reasonable bounds

        # Optimize the model
        gurobi_model.optimize()

        # Check if the model is infeasible
        if gurobi_model.Status in [gp.GRB.INFEASIBLE, gp.GRB.INF_OR_UNBD]:
            print("Model is infeasible. Computing IIS...")

            # Compute the IIS
            gurobi_model.computeIIS()

            # Write the IIS to a file
            gurobi_model.write("model.ilp")

            # Read and print the IIS file
            with open("model.ilp", "r") as iis_file:
                iis = iis_file.read()
            print("IIS:\n", iis)

        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("pyomo.util.infeasible")
        # solving and check infeasibility
        min_gap = 0.0001
        opt = SolverFactory("gurobi")
        opt.options["mipgap"] = min_gap
        opt.options["TimeLimit"] = 15 * 60  # in seconds
        opt.options["MIPFocus"] = 3  # focus on improving the bound
        opt.options["Nonconvex"] = 2  # non-convex problem
        opt.options["Threads"] = 20  # use CPU more efficiently, not optimized yet
        opt.options["BarHomogeneous"] = 1
        opt.options["OptimalityTol"] = 1e-6
        opt.options["FeasibilityTol"] = 1e-6
        opt.options["IntFeasTol"] = 1e-6

        opt.options["DualReductions"] = (
            0  # set to see is either infeasible or unbounded
        )
        results = opt.solve(self.model, tee=True)

        # Check solver status
        if results.solver.termination_condition == "infeasible":
            logger.info("The model is infeasible. Logging infeasible constraints...")
            log_infeasible_constraints(
                self.model, log_expression=True, log_variables=True
            )
        else:
            logger.info(
                "The model is feasible. Objective value: {}".format(
                    value(self.model.objective)
                )
            )

        from pyomo.core.expr.visitor import identify_variables

        all_vars = set()
        for v in self.model.component_objects(Var, active=True):
            for index in v:
                all_vars.add((v, index))

        # Collect variables that appear in constraints
        constrained_vars = set()
        for c in self.model.component_objects(Constraint, active=True):
            for index in c:
                for var in identify_variables(c[index].body):
                    constrained_vars.add((var.parent_component(), var.index()))

        # Find unconstrained variables
        unconstrained_vars = all_vars - constrained_vars

        # Display unconstrained variables
        if unconstrained_vars:
            print("Unconstrained Variables:")
            for var, index in unconstrained_vars:
                print(f"  {var.name}[{index}]")
        else:
            print("All variables are constrained.")

        tolerance = 1e-6  # Define a small tolerance for floating-point comparisons

        try:  # Loop through each constraint
            for constraint in self.model.component_objects(Constraint, active=True):
                print(f"Checking constraint: {constraint.name}")
                for index in constraint:
                    # Evaluate LHS and RHS
                    lhs_value = value(constraint[index].body)
                    rhs_value = value(constraint[index].upper)
                    lhs_lb_value = value(constraint[index].lower)

                    # Check if the LHS exceeds the upper bound
                    if rhs_value is not None and lhs_value > rhs_value + tolerance:
                        print(
                            f"Violation at {index}: LHS = {lhs_value}, RHS = {rhs_value}"
                        )

                    # Check if the LHS is below the lower bound
                    if (
                        lhs_lb_value is not None
                        and lhs_value < lhs_lb_value - tolerance
                    ):
                        print(
                            f"Violation at {index}: LHS = {lhs_value}, LB = {lhs_lb_value}"
                        )
        except:
            print("No constraints violation")

    def comparison(self, pp_dynamic_losses):
        """
        Compares the results of the Second-Order Conic Programming (SOCP) model with the results from Pandapower for power losses in the grid.
        If the mode is 'static', it visualizes the losses over the lines and calculates the percentage difference between SOCP and Pandapower results.
        If the mode is not 'static', it compares the SOCP results with the dynamic losses provided by Pandapower.
        """

        #  comparison
        MW_to_kW = 1000

        if self.mode == "static":
            # pandapower
            pp_results = self.net.res_bus
            pp_results["Bus name"] = [self.bus_dict[i] for i in pp_results.index]

            # pyomo model results
            res_SOCP = pd.DataFrame(
                self.model.P_loss.extract_values().items(),
                columns=["index", "P_loss_kW"],
            )
            res_SOCP_res_bus = pd.DataFrame(
                self.model.V_m_sqr.extract_values().items(),
                columns=["index", "Voltage_m_sqr"],
            )
            res_SOCP_res_bus["Bus"] = res_SOCP_res_bus.apply(
                lambda row: row["index"][0], axis=1
            )

            bus_dict = dict(zip(self.net.bus.name, self.net.bus.index))
            res_SOCP_res_bus["Bus_num"] = [bus_dict[i] for i in res_SOCP_res_bus["Bus"]]

            res_SOCP_res_bus["time"] = res_SOCP_res_bus.apply(
                lambda row: row["index"][1], axis=1
            )

            res_SOCP_res_bus["vm_pu"] = res_SOCP_res_bus.apply(
                lambda row: np.sqrt(row["Voltage_m_sqr"]), axis=1
            )

            res_SOCP_res_bus.set_index("Bus_num", inplace=True)

            res_SOCP_res_bus["net_p_kW"] = self.model.net_p.extract_values().values()
            res_SOCP_res_bus["net_q_kVar"] = self.model.net_q.extract_values().values()

            pp_res = self.net.res_bus
            lv_trafo_bus_n = self.net.trafo.lv_bus.iloc[0]
            pp_res.at[lv_trafo_bus_n, "p_mw"] = -self.net.res_ext_grid.p_mw.iloc[
                0
            ]  # changed here, because change of the slack bus
            pp_res.at[lv_trafo_bus_n, "q_mvar"] = -self.net.res_ext_grid.q_mvar.iloc[0]
            pp_res = pp_res[
                pp_res.index.isin(res_SOCP_res_bus.index)
            ]  # getting same indexes/buses of the SOCP for comparison

            # changing the trafo lv results into the bus as the SOCP can't consider the trafo characteristics, only buses

            # p-comparison plot
            _, ax = plt.subplots()
            pp_non_null = pp_res[pp_res["p_mw"] != 0]
            res_SOCP_non_null = res_SOCP_res_bus[res_SOCP_res_bus["net_p_kW"] != 0]

            # res_SOCP_non_null.index == pp_non_null.index

            ax.scatter(
                pp_non_null["Bus name"],
                -pp_non_null["p_mw"] * MW_to_kW,
                edgecolors="k",
                alpha=0.5,
                label="Pandpower [kW]",
            )  # negative because the convention is that is positive if it exports
            ax.scatter(
                res_SOCP_non_null["Bus"],
                res_SOCP_non_null["net_p_kW"],
                edgecolors="k",
                alpha=0.5,
                label="SOCP [W]",
            )

            plt.axhline(
                y=0,
                color="k",
                linestyle="--",
                label="Generation (>0) / consumption (<0)",
            )

            plt.xticks(rotation=45, ha="right")
            plt.legend()
            plt.ylabel("P [kW]")
            plt.title("Comparison in P")

            # q-comparison plot
            _, ax = plt.subplots()
            pp_non_null = pp_res[pp_res["q_mvar"] != 0]
            res_SOCP_non_null = res_SOCP_res_bus[res_SOCP_res_bus["net_q_kVar"] != 0]

            # res_SOCP_non_null.index == pp_non_null.index

            ax.scatter(
                pp_non_null["Bus name"],
                -pp_non_null["q_mvar"] * MW_to_kW,
                edgecolors="k",
                alpha=0.5,
                label="Pandpower [kW]",
            )  # negative because the convention is that is positive if it exports
            ax.scatter(
                res_SOCP_non_null["Bus"],
                res_SOCP_non_null["net_q_kVar"],
                edgecolors="k",
                alpha=0.5,
                label="SOCP [W]",
            )

            plt.axhline(
                y=0,
                color="k",
                linestyle="--",
                label="Generation (>0) / consumption (<0)",
            )

            plt.xticks(rotation=45, ha="right")
            plt.legend(loc="lower left")
            plt.ylabel("Q [kVAr]")
            plt.title("Comparison in Q")

            # V-comparison plot
            _, ax = plt.subplots(figsize=(15, 6))

            ax.scatter(
                pp_res["Bus name"],
                pp_res["vm_pu"],
                edgecolors="k",
                alpha=0.5,
                label="Pandpower [kW]",
            )  # negative because the convention is that is positive if it exports
            ax.scatter(
                res_SOCP_res_bus["Bus"],
                res_SOCP_res_bus["vm_pu"],
                edgecolors="k",
                alpha=0.5,
                label="SOCP [kW]",
            )
            plt.axhline(y=0.95, color="k", linestyle="--")
            plt.axhline(y=1.05, color="k", linestyle="--")

            plt.xticks(rotation=45, ha="right")
            plt.legend()
            plt.ylabel("V_pu [-]")
            plt.title("Comparison in V_pu")
            plt.show()

            # logging info
            print(f"{self.net_name} \n")
            print("SOCP solving grid with loss of kW \n")
            print(res_SOCP["P_loss_kW"].unique().sum())

            print("Pandapower solving with loss of kW \n")
            print(self.total_active_power_loss * MW_to_kW)

            res_SOCP["line_name"] = res_SOCP.apply(lambda row: row["index"][0], axis=1)
            res_SOCP.set_index("line_name", inplace=True)
            res_SOCP.sort_index(inplace=True)

            line_dict = dict(zip(self.net.line.index, self.net.line.name))
            pp_df = pd.DataFrame()

            pp_df["losses_kW"] = self.net.res_line.pl_mw * MW_to_kW
            pp_df["line_name"] = [line_dict[i] for i in pp_df.index]
            pp_df.set_index("line_name", inplace=True)
            pp_df.sort_index(inplace=True)

            plt.figure()
            res_SOCP["P_loss_kW"].plot(label="SOCP")
            pp_df["losses_kW"].plot(label="Pandapower")
            plt.xticks(rotation=45)
            plt.title("Losses over the lines")
            plt.xlabel("Lines name")
            plt.ylabel("Loss [W]")
            plt.legend()
            plt.tight_layout()
            plt.show()

            diff_SOCP_pp = (
                100
                * abs(
                    res_SOCP["P_loss_kW"].unique().sum()
                    - (self.total_active_power_loss * MW_to_kW)
                )
                / (self.total_active_power_loss * MW_to_kW)
            )
            print(f"difference of {diff_SOCP_pp} %")
            self.SOCP_res_bus = res_SOCP_res_bus

            return res_SOCP_res_bus["vm_pu"]

        else:
            # todo: implment also the bus part for the dynamic simulation?
            res_SOCP = pd.DataFrame(
                self.model.P_loss.extract_values().items(),
                columns=["index", "P_loss_W"],
            )
            print(f"{self.net_name} \n")
            print("SOCP solving grid with loss of kW \n")
            print(res_SOCP["P_loss_kW"].unique().sum())
            # pp_dynamic_losses
            print("Pandapower solving with loss of kW \n")
            print(pp_dynamic_losses)

            diff_SOCP_pp = (
                100
                * abs(res_SOCP["P_loss_kW"].unique().sum() - (pp_dynamic_losses))
                / (pp_dynamic_losses)
            )
            print(f"difference of {diff_SOCP_pp} %")

            return

    def getting_V_from_SOCP(self):

        res_SOCP_res_bus = pd.DataFrame(
            self.model.V_m_sqr.extract_values().items(),
            columns=["index", "Voltage_m_sqr"],
        )
        res_SOCP_res_bus["Bus"] = res_SOCP_res_bus.apply(
            lambda row: row["index"][0], axis=1
        )

        bus_dict = dict(zip(self.net.bus.name, self.net.bus.index))
        res_SOCP_res_bus["Bus_num"] = [bus_dict[i] for i in res_SOCP_res_bus["Bus"]]

        res_SOCP_res_bus["time"] = res_SOCP_res_bus.apply(
            lambda row: row["index"][1], axis=1
        )

        res_SOCP_res_bus["vm_pu"] = res_SOCP_res_bus.apply(
            lambda row: np.sqrt(row["Voltage_m_sqr"]), axis=1
        )

        res_SOCP_res_bus.set_index("Bus_num", inplace=True)

        res_SOCP_res_bus["net_p_kW"] = self.model.net_p.extract_values().values()
        res_SOCP_res_bus["p_import_kW"] = self.model.p_import.extract_values().values()
        res_SOCP_res_bus["p_export_kW"] = self.model.p_export.extract_values().values()
        res_SOCP_res_bus["net_q_kVar"] = self.model.net_q.extract_values().values()

        return res_SOCP_res_bus, res_SOCP_res_bus["vm_pu"]

    def Plotting_static_simulation(self):

        bus_dict_rev = dict(zip(self.net.bus.name, self.net.bus.index))

        fig, ax = plt.subplots()
        for index, line in self.net.line_geodata.iterrows():
            Xs = [coord[0] for coord in line[0]]
            ys = [coord[1] for coord in line[0]]
            ax.plot(Xs, ys, color="k")

        self.SOCP_res_bus["x"] = self.SOCP_res_bus.apply(
            lambda row: self.net.bus_geodata.loc[bus_dict_rev[row.Bus], "x"], axis=1
        )
        self.SOCP_res_bus["y"] = self.SOCP_res_bus.apply(
            lambda row: self.net.bus_geodata.loc[bus_dict_rev[row.Bus], "y"], axis=1
        )

        max_size = 800
        abs_net_p = self.SOCP_res_bus["net_p_kW"].abs()

        if abs_net_p.max() - abs_net_p.min() > 1:  # 1 kW
            sizes = 50 + max_size * (abs_net_p / abs_net_p.max())

            sc = ax.scatter(
                self.SOCP_res_bus["x"],
                self.SOCP_res_bus["y"],
                s=sizes,
                marker="o",
                c=self.SOCP_res_bus["vm_pu"],
                cmap="rainbow",
                edgecolors="k",
            )
            plt.colorbar(sc, ax=ax, label="Vm_pu")

            legend_sizes = np.linspace(abs_net_p.min(), abs_net_p.max(), 5)

            for size in legend_sizes:
                ax.scatter(
                    [],
                    [],
                    s=max_size * abs(size / abs_net_p.max()),
                    c="k",
                    alpha=0.6,
                    label=f"{int(size)} kW",
                )

            ax.legend(
                title="Active power (kW)",
                scatterpoints=1,
                loc=0,
                ncol=6,
                fontsize=8,
                handleheight=2,
                labelspacing=2.5,
                borderpad=1,
            )
            ax.get_xaxis().set_visible(False)

            # Hide Y-axis
            ax.get_yaxis().set_visible(False)
            for spine in ax.spines.values():
                spine.set_visible(False)

        else:

            sc = ax.scatter(
                self.SOCP_res_bus["x"],
                self.SOCP_res_bus["y"],
                marker="o",
                c=self.SOCP_res_bus["vm_pu"],
                cmap="rainbow",
                edgecolors="k",
            )
            plt.colorbar(sc, ax=ax, label="Vm_pu")

            ax.get_xaxis().set_visible(False)

            # Hide Y-axis
            ax.get_yaxis().set_visible(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
