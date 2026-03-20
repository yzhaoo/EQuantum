# Self-Consistent Quantum–Poisson Solver (FSC)

## Overview

This project implements a **self-consistent loop** coupling:

- **Quantum solver (KPM-based LDOS)**
- **Electrostatics (Poisson solver)**
- **Local nonlinear solver (charge consistency)**

The goal is to simulate quantum systems (e.g., GaAs 2DEG) under gating, including **depletion effects**, while maintaining numerical stability and efficiency.

---

# 🧠 Core Idea

We solve for a self-consistent solution of:

- Local potential: `Ui`
- Local charge density: `ni`
- Local density of states: `LDOS(E)`

under the condition:

\[
n_i = \int_{-\infty}^{\mu} \rho_i(E - U_i)\, dE
\]

coupled with Poisson:

\[
\Delta U = C_i^{-1} (n - n_{\text{ext}})
\]

---

# 🔄 Algorithm Workflow

## Initialization

1. Build full system geometry
2. Assign materials (`Qsystem`, `gate`, `dielectric`, etc.)
3. Initialize:
   - `Ui` (potential)
   - `ni` (charge)
4. Solve initial Poisson problem
5. Compute initial LDOS

---

## Main Iteration Loop

Repeat until convergence:

### Step 1 — Local Solver

For each active site:

- Solve:
  \[
  dU \cdot C_i + n_i = \int \rho_i(E - U_i)\, dE
  \]

- Update:
  - `Ui += dU`
  - `ni += dn`

---

### Step 2 — Update Active Region (`Qprime`)

Define active quantum region:

- Sites with significant LDOS / charge response
- Excludes strongly depleted regions

---

### Step 3 — Rebuild Quantum System (NEW)

If `Qprime` changes:

- Rebuild Hamiltonian **only on Qprime**
- Update mappings:
  - `site_id → Hamiltonian index`
- Recompute Poisson coupling (`Ci`)

---

### Step 4 — Poisson Update

Solve electrostatics:

- Update `Ui` and `ni` globally

---

### Step 5 — Quantum Update

- Build Hamiltonian on **Qprime only**
- Compute LDOS using KPM
- Map results back to global structure

---

# ⚡ Key Optimization: Active Region Hamiltonian

## Problem

Including all sites in Hamiltonian leads to:

- Huge onsite potentials (depleted regions)
- Very large spectral width
- KPM resolution collapse

## Solution

Use **active-region Hamiltonian**:

- Only include sites in `Qprime`
- Remove depleted regions from quantum problem

### Result

✔ Stable energy bounds  
✔ High-resolution KPM  
✔ Faster computation  

---

# 🧩 Data Structure Design

## Global (FSC / System)

- `Qsites` → all quantum-capable site IDs
- `Qprime` → active subset
- `Ui`, `ni` → full arrays

## Quantum System

- `Qsite_ids` → active sites only
- `q_to_Q_map` → site_id → matrix index

## Mapping

- `Qp_in_Q` → maps active indices back to global indices

---

# 📊 LDOS Handling

## Before

- LDOS computed for all sites

## Now

- LDOS computed **only for Qprime**
- Written back into full array:

```python
self.ildos[global_idx] = new_ildos[local_idx]# Self-Consistent Quantum–Poisson Solver (FSC)

## Overview

This project implements a **self-consistent loop** coupling:

- **Quantum solver (KPM-based LDOS)**
- **Electrostatics (Poisson solver)**
- **Local nonlinear solver (charge consistency)**

The goal is to simulate quantum systems (e.g., GaAs 2DEG) under gating, including **depletion effects**, while maintaining numerical stability and efficiency.

---

# 🧠 Core Idea

We solve for a self-consistent solution of:

- Local potential: `Ui`
- Local charge density: `ni`
- Local density of states: `LDOS(E)`

under the condition:

\[
n_i = \int_{-\infty}^{\mu} \rho_i(E - U_i)\, dE
\]

coupled with Poisson:

\[
\Delta U = C_i^{-1} (n - n_{\text{ext}})
\]

---

# 🔄 Algorithm Workflow

## Initialization

1. Build full system geometry
2. Assign materials (`Qsystem`, `gate`, `dielectric`, etc.)
3. Initialize:
   - `Ui` (potential)
   - `ni` (charge)
4. Solve initial Poisson problem
5. Compute initial LDOS

---

## Main Iteration Loop

Repeat until convergence:

### Step 1 — Local Solver

For each active site:

- Solve:
  \[
  dU \cdot C_i + n_i = \int \rho_i(E - U_i)\, dE
  \]

- Update:
  - `Ui += dU`
  - `ni += dn`

---

### Step 2 — Update Active Region (`Qprime`)

Define active quantum region:

- Sites with significant LDOS / charge response
- Excludes strongly depleted regions

---

### Step 3 — Rebuild Quantum System (NEW)

If `Qprime` changes:

- Rebuild Hamiltonian **only on Qprime**
- Update mappings:
  - `site_id → Hamiltonian index`
- Recompute Poisson coupling (`Ci`)

---

### Step 4 — Poisson Update

Solve electrostatics:

- Update `Ui` and `ni` globally

---

### Step 5 — Quantum Update

- Build Hamiltonian on **Qprime only**
- Compute LDOS using KPM
- Map results back to global structure

---

# ⚡ Key Optimization: Active Region Hamiltonian

## Problem

Including all sites in Hamiltonian leads to:

- Huge onsite potentials (depleted regions)
- Very large spectral width
- KPM resolution collapse

## Solution

Use **active-region Hamiltonian**:

- Only include sites in `Qprime`
- Remove depleted regions from quantum problem

### Result

✔ Stable energy bounds  
✔ High-resolution KPM  
✔ Faster computation  

---

# 🧩 Data Structure Design

## Global (FSC / System)

- `Qsites` → all quantum-capable site IDs
- `Qprime` → active subset
- `Ui`, `ni` → full arrays

## Quantum System

- `Qsite_ids` → active sites only
- `q_to_Q_map` → site_id → matrix index

## Mapping

- `Qp_in_Q` → maps active indices back to global indices

---

# 📊 LDOS Handling

## Before

- LDOS computed for all sites

## Now

- LDOS computed **only for Qprime**
- Written back into full array:

```python
self.ildos[global_idx] = new_ildos[local_idx]