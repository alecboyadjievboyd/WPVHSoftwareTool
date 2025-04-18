import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import pulp
import math

from pulp import PULP_CBC_CMD
print(PULP_CBC_CMD().path)
# -------------------------------
# Helper Function to Check Weights
# -------------------------------
def check_weights(weights):
    if len(weights) != 5:
        raise ValueError("Weights must be a list of 5 numbers.")
    if abs(sum(weights) - 1) > 1e-6:
        raise ValueError("Weights must sum to 1.")

# === PRODUCT (Multiplicative) Utility Models ===
import pulp
import math

# -------------------------------
# Helper Function to Check Weights
# -------------------------------
def check_weights(weights):
    if len(weights) != 5:
        raise ValueError("Weights must be a list of 5 numbers.")
    if abs(sum(weights) - 1) > 1e-6:
        raise ValueError("Weights must sum to 1.")

# ====================================
# PRODUCT (Multiplicative) Utility Models
# ====================================

def build_product_model(model_type, C, options, policies, weights, num_breakpoints=10):
    """
    Builds a MILP using the multiplicative (product) utility function.
    
    The weighted utility for state j is:
       U(t,j) = sum_{d=0}^{4} [ weight[d] * (S_j[d] * (prod_{i in I_d}(1+t_i*r_i[d])) ) ]
    
    We linearize the product via a logarithmic transformation and piecewise linear approximation.
    """
    check_weights(weights)
    num_options = len(options)
    num_aspects = 5  # There are 5 SPACE aspects.
    
    prob = pulp.LpProblem(f"Product_{model_type}", pulp.LpMaximize)
    
    # Decision variables: prod_t_i for each option (binary)
    t = [pulp.LpVariable(f"prod_t_{i}", cat="Binary") for i in range(num_options)]
    
    # Budget constraint
    prob += pulp.lpSum(options[i]["cost"] * t[i] for i in range(num_options)) <= C, "Budget"



    
    # For each aspect d, set up the linearization of the multiplicative effect.
    prod_x_vars = [None] * num_aspects   # x_d = sum_{i in I_d} t_i * log(1 + r_i[d])
    prod_z_vars = [None] * num_aspects   # z_d approximates exp(x_d)
    prod_lam_vars = [None] * num_aspects # lambda variables for piecewise approximation
    
    for d in range(num_aspects):
        # Find options affecting aspect d.
        I_d = [i for i in range(num_options) if options[i]["improvement"][d] > 0]
        if I_d:
            x_d = pulp.LpVariable(f"prod_x_{d}", lowBound=0, cat="Continuous")
            prod_x_vars[d] = x_d
            # Define x_d exactly as the sum of log factors over options affecting aspect d.
            prob += x_d == pulp.lpSum(math.log(1 + options[i]["improvement"][d]) * t[i] for i in I_d), f"def_prod_x_{d}"
            
            # Determine the range for x_d and set up breakpoints.
            x_min = 0
            x_max = sum(math.log(1 + options[i]["improvement"][d]) for i in I_d)
            x_bp = [x_min + k*(x_max - x_min)/(num_breakpoints - 1) for k in range(num_breakpoints)]
            z_bp = [math.exp(x) for x in x_bp]
            
            # Lambda variables for the convex combination.
            lam_d = [pulp.LpVariable(f"prod_lam_{d}_{k}", lowBound=0, upBound=1, cat="Continuous")
                     for k in range(num_breakpoints)]
            prod_lam_vars[d] = lam_d
            prob += pulp.lpSum(lam_d[k] for k in range(num_breakpoints)) == 1, f"prod_lam_sum_{d}"
            prob += x_d == pulp.lpSum(lam_d[k]*x_bp[k] for k in range(num_breakpoints)), f"prod_lam_x_{d}"
            z_d = pulp.LpVariable(f"prod_z_{d}", lowBound=1, cat="Continuous")  # Lower bound 1 since exp(0)=1.
            prod_z_vars[d] = z_d
            prob += z_d == pulp.lpSum(lam_d[k]*z_bp[k] for k in range(num_breakpoints)), f"prod_lam_z_{d}"
        else:
            prod_x_vars[d] = None
            prod_z_vars[d] = 1  # No improvement possible on aspect d.
            prod_lam_vars[d] = None
    
    # Define state utilities for the product model using weighted average.
    product_state_utilities = []
    for j, state in enumerate(policies):
        U_j = sum(weights[d] * (state["S"][d] * (prod_z_vars[d] if isinstance(prod_z_vars[d], pulp.LpVariable) else prod_z_vars[d]))
                  for d in range(num_aspects))
        product_state_utilities.append(U_j)
    
    # Set up the objective based on the decision paradigm.
    if model_type == "MEU":
        # Maximize expected utility: sum_j p_j * U(t,j).
        # Precompute weighted average S:
        S_avg = [sum(state["p"] * state["S"][d] for state in policies) for d in range(num_aspects)]
        obj_expr = sum(weights[d] * S_avg[d] * (prod_z_vars[d] if isinstance(prod_z_vars[d], pulp.LpVariable) else prod_z_vars[d])
                       for d in range(num_aspects))
        prob += obj_expr, "Expected_Product_Utility"
    elif model_type == "MaxiMin":
        # Maximize the worst-case utility.
        w = pulp.LpVariable("prod_w", lowBound=0, cat="Continuous")
        for j, U_j in enumerate(product_state_utilities):
            prob += w <= U_j, f"prod_min_state_{j}"
        prob += w, "Worst_Product_Utility"
    elif model_type == "MaxiMax":
        # Maximize the best-case utility by selecting one state.
        w = pulp.LpVariable("prod_w", lowBound=0, cat="Continuous")
        k_states = len(policies)
        y = [pulp.LpVariable(f"prod_y_{j}", cat="Binary") for j in range(k_states)]
        prob += pulp.lpSum(y[j] for j in range(k_states)) == 1, "prod_one_state_selected"
        M_val = 1e4  # Big-M constant.
        for j, U_j in enumerate(product_state_utilities):
            prob += w <= U_j + M_val*(1 - y[j]), f"prod_max_state_{j}"
        prob += w, "Best_Product_Utility"
    else:
        raise ValueError("Unknown model type")
    
    return prob, t, prod_x_vars, prod_z_vars, prod_lam_vars

def optimize_product_meu(C, options, policies, weights, num_breakpoints=10):
    model, t, x_vars, z_vars, lam_vars = build_product_model("MEU", C, options, policies, weights, num_breakpoints)
    model.solve()
    selected = [i for i, var in enumerate(t) if pulp.value(var) > 0.5]
    obj_val = pulp.value(model.objective)
    return selected, obj_val

def optimize_product_maximin(C, options, policies, weights, num_breakpoints=10):
    model, t, x_vars, z_vars, lam_vars = build_product_model("MaxiMin", C, options, policies, weights, num_breakpoints)
    model.solve()
    selected = [i for i, var in enumerate(t) if pulp.value(var) > 0.5]
    w_val = pulp.value(model.variablesDict().get("prod_w"))
    return selected, w_val

def optimize_product_maximax(C, options, policies, weights, num_breakpoints=10):
    model, t, x_vars, z_vars, lam_vars = build_product_model("MaxiMax", C, options, policies, weights, num_breakpoints)
    model.solve()
    selected = [i for i, var in enumerate(t) if pulp.value(var) > 0.5]
    w_val = pulp.value(model.variablesDict().get("prod_w"))
    return selected, w_val

# ====================================
# SUM (Additive) Utility Models
# ====================================

def build_sum_model(model_type, C, options, policies, weights):
    """
    Builds a MILP using the additive (sum) utility function.
    
    The weighted utility for state j is:
       U(t,j) = sum_{d=0}^{4} [ weight[d] * ( S_j[d] + sum_{i=1}^n t_i * r_i[d] ) ]
    
    This formulation is completely linear.
    """
    check_weights(weights)
    num_options = len(options)
    num_aspects = 5
    
    prob = pulp.LpProblem(f"Sum_{model_type}", pulp.LpMaximize)
    
    # Decision variables: sum_t_i for each option.
    t = [pulp.LpVariable(f"sum_t_{i}", cat="Binary") for i in range(num_options)]
    
    # Budget constraint.
    prob += pulp.lpSum(options[i]["cost"] * t[i] for i in range(num_options)) <= C, "Budget"

    # Budget constraint.

# Enforce selecting only one option.

    
    # For each aspect, compute the additive improvement.
    improvement_aspect = {}
    for d in range(num_aspects):
        improvement_aspect[d] = pulp.lpSum(t[i] * options[i]["improvement"][d] for i in range(num_options))
    
    # Define state utilities for the sum model using the weighted average.
    sum_state_utilities = []
    for j, state in enumerate(policies):
        U_j = sum(weights[d] * (state["S"][d] + improvement_aspect[d]) for d in range(num_aspects))
        sum_state_utilities.append(U_j)
    
    # Set the objective based on the model type.
    if model_type == "MEU":
        # Expected utility: sum_j p_j * U(t,j).
        S_avg = [sum(state["p"] * state["S"][d] for state in policies) for d in range(num_aspects)]
        # The constant part plus the variable part:
        obj_expr = sum(weights[d] * S_avg[d] for d in range(num_aspects)) + sum(weights[d] * improvement_aspect[d] for d in range(num_aspects))
        prob += obj_expr, "Expected_Sum_Utility"
    elif model_type == "MaxiMin":
        w = pulp.LpVariable("sum_w", lowBound=0, cat="Continuous")
        for j, U_j in enumerate(sum_state_utilities):
            prob += w <= U_j, f"sum_min_state_{j}"
        prob += w, "Worst_Sum_Utility"
    elif model_type == "MaxiMax":
        w = pulp.LpVariable("sum_w", lowBound=0, cat="Continuous")
        k_states = len(policies)
        y = [pulp.LpVariable(f"sum_y_{j}", cat="Binary") for j in range(k_states)]
        prob += pulp.lpSum(y[j] for j in range(k_states)) == 1, "sum_one_state_selected"
        M_val = 1e4
        for j, U_j in enumerate(sum_state_utilities):
            prob += w <= U_j + M_val*(1 - y[j]), f"sum_max_state_{j}"
        prob += w, "Best_Sum_Utility"
    else:
        raise ValueError("Unknown model type")
    
    return prob, t

def optimize_sum_meu(C, options, policies, weights):
    model, t = build_sum_model("MEU", C, options, policies, weights)
    model.solve()
    selected = [i for i, var in enumerate(t) if pulp.value(var) > 0.5]
    obj_val = pulp.value(model.objective)
    return selected, obj_val

def optimize_sum_maximin(C, options, policies, weights):
    model, t = build_sum_model("MaxiMin", C, options, policies, weights)
    model.solve()
    selected = [i for i, var in enumerate(t) if pulp.value(var) > 0.5]
    w_val = pulp.value(model.variablesDict().get("sum_w"))
    return selected, w_val

def optimize_sum_maximax(C, options, policies, weights):
    model, t = build_sum_model("MaxiMax", C, options, policies, weights)
    model.solve()
    selected = [i for i, var in enumerate(t) if pulp.value(var) > 0.5]
    w_val = pulp.value(model.variablesDict().get("sum_w"))
    return selected, w_val

# === GUI APP ===
class OptimizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Optimization UI")
        
        self.weights = [tk.StringVar(value="0.2") for _ in range(5)]
        self.budget = tk.StringVar(value="100")
        self.excel_path = tk.StringVar()
        self.result_text = tk.Text(self.root, height=25, width=100)

        self.setup_ui()

    def setup_ui(self):
        tk.Label(self.root, text="Budget:").grid(row=0, column=0, sticky="e")
        tk.Entry(self.root, textvariable=self.budget).grid(row=0, column=1, sticky="w")

        labels = ["Satisfaction", "Performance", "Activity", "Communication", "Efficiency"]
        for i in range(5):
            tk.Label(self.root, text=f"Weight ({labels[i]}):").grid(row=i+1, column=0, sticky="e")
            tk.Entry(self.root, textvariable=self.weights[i]).grid(row=i+1, column=1, sticky="w")

        tk.Button(self.root, text="Upload Excel File", command=self.load_excel).grid(row=6, column=0, pady=10)
        tk.Label(self.root, textvariable=self.excel_path, wraplength=400).grid(row=6, column=1)
        tk.Button(self.root, text="Run Optimization", command=self.run_optimization).grid(row=7, column=0, columnspan=2, pady=10)
        self.result_text.grid(row=8, column=0, columnspan=2, pady=10)

    def load_excel(self):
        path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
        if path:
            self.excel_path.set(path)

    def check_policy_probabilities(self, policies):
        total_prob = sum(p["p"] for p in policies)
        if abs(total_prob - 1) > 1e-6:
            raise ValueError(f"Policy probabilities must sum to 1. Got {total_prob:.6f} instead.")

    def run_optimization(self):
        try:
            C = float(self.budget.get())
            weights = [float(w.get()) for w in self.weights]
            check_weights(weights)

            df = pd.read_excel(self.excel_path.get(), sheet_name=None)
            options_df = df['Options']
            options_df.columns = options_df.columns.str.strip()

            for i in range(5):
                if f'improvement[{i}]' not in options_df.columns:
                    options_df.rename(columns={f'improvement[]': f'improvement[{i}]'}, inplace=True)

            expected_cols = ["cost"] + [f"improvement[{i}]" for i in range(5)]
            options_df = options_df[expected_cols]
            policies_df = df['Policies']

            options = [
                {
                    "cost": row["cost"],
                    "improvement": [row[f"improvement[{i}]"] for i in range(5)]
                }
                for _, row in options_df.iterrows()
            ]

            policies = [
                {
                    "p": row["p"],
                    "S": [row[f"S[{i}]"] for i in range(5)]
                }
                for _, row in policies_df.iterrows()
            ]
            self.check_policy_probabilities(policies)

            results = []

            results.append("=== PRODUCT MODELS ===")
            for label, func in [
                ("MEU", optimize_product_meu),
                ("MaxiMin", optimize_product_maximin),
                ("MaxiMax", optimize_product_maximax),
            ]:
                selected, val = func(C, options, policies, weights)
                formatted_selection = ", ".join([f"Option {s}" for s in selected])
                results.append(f"{label} Selected: {formatted_selection} | Value: {val:.2f}")

            results.append("\n=== SUM MODELS ===")
            for label, func in [
                ("MEU", optimize_sum_meu),
                ("MaxiMin", optimize_sum_maximin),
                ("MaxiMax", optimize_sum_maximax),
            ]:
                selected, val = func(C, options, policies, weights)
                formatted_selection = ", ".join([f"Option {s}" for s in selected])
                results.append(f"{label} Selected: {formatted_selection} | Value: {val:.2f}")

            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(tk.END, "\n".join(results))

        except Exception as e:
            messagebox.showerror("Error", str(e))


# Start the app
if __name__ == "__main__":
    root = tk.Tk()
    app = OptimizationApp(root)
    root.mainloop()
