# Architecture Design

## Overview
This project implements the LAM (Linearized Active-Set Method) / StQP model for portfolio optimization as described in Cesarone et al. (2011).
The system takes historical stock price data, calculates expected returns and covariance, and solves the portfolio optimization problem under cardinality and quantity constraints.

## System Components

### 1. Data Layer (`src/data_loader.py`)
Responsible for loading and preprocessing the financial data.
- **Input**: CSV file (`data/topix-large500.csv`)
- **Responsibilities**:
    - Load CSV with correct encoding (Shift-JIS).
    - Clean data (skip header garbage).
    - Handle missing values.
    - Calculate daily returns.
    - Compute expected return vector ($\mu$) and covariance matrix ($\Sigma$).

### 2. Optimization Engine (`src/solver.py`)
Implements the core optimization logic.
- **Input**: $\mu, \Sigma$, target return $\rho$, cardinality $K$, bounds $\ell, u$, penalty $M$.
- **Algorithm**: Increasing Set Algorithm (IS).
    - **Outer Loop**: Iterate cardinality $k$ from 1 to $K$.
    - **Inner Logic**:
        - Select candidate sets of assets.
        - Solve the quadratic programming (QP) problem for the specific subset.
        - Check KKT conditions and constraints.
        - Use a QP solver (e.g., `cvxpy` or `scipy.optimize`) for the sub-problems.

### 3. Application Entry Point (`main.py`)
Orchestrates the process.
- Load data.
- Define parameters.
- Call the solver.
- Display/Save results (optimal weights, portfolio return, risk).

## Data Flow
1. `main.py` calls `data_loader.load_data()`.
2. `data_loader` returns $\mu, \Sigma$.
3. `main.py` calls `solver.solve_lam_stqp(\mu, \Sigma, ...)`
4. `solver` returns optimal $x$.
5. `main.py` outputs results.

## Dependencies
- Python 3.x
- pandas (Data manipulation)
- numpy (Linear algebra)
- scipy (Optimization)
- cvxpy (Optional, for easier QP formulation)
