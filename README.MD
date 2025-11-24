# Climate Economics Lectures

This repository contains all teaching materials for two lectures of four hours each on climate economics, developed and taught by **Gauthier Vermandel (CMAP, Ecole Polytechnique)**.

The course introduces the physical and economic foundations of climate change, and provides hands on sessions using Python to simulate climate systems and Integrated Assessment Models (IAMs).

The repository is organized into two main folders:

- **`Lecture_1_climate_and_eco_separate/`**  
  Climate science, greenhouse gas dynamics, and macroeconomic foundations.

- **`Lecture_2_integrated_assessment_models/`**  
  Construction, calibration, and simulation of IAMs, including the DICE model.

---

## Lecture 1: Climate and Economics (separate blocks)

**Folder:** `Lecture_1_climate_and_eco_separate/`

### Contents

- **Notebook_ClimateModels.ipynb**  
  Interactive notebook introducing simplified climate and carbon cycle models.

- **Climate_Models.py**  
  Python implementation of a two box temperature model and a three reservoir carbon system.

- **Slides_0A.pdf** and **Slides_0B.pdf**  
  Slides covering climate physics, greenhouse gases, radiative forcing, and links with macroeconomic variables.

- **OptimalGrowth.py** and **Notebook_OptimalGrowth.ipynb**  
  Ramsey style growth model used to introduce welfare, dynamic decisions, and externalities.

- **growth_model.pdf**  
  Short note introducing key economic growth concepts relevant to climate economics.

### Learning objectives

- Understand greenhouse gases, radiative forcing, and temperature dynamics.  
- Run and explore simple climate models in Python.  
- Link macroeconomic quantities with climate processes.  
- Prepare for integrated models that merge climate science and economics.

---

## Lecture 2: Integrated Assessment Models

**Folder:** `Lecture_2_integrated_assessment_models/`

### Contents

- **DICE.py**  
  Python implementation of the DICE model.

- **DICE_Notebook1-1_basics.ipynb**  
  Notebook explaining the IAM structure and the components of DICE.

- **DICE_Notebook1-2_generation_scenarios.ipynb**  
  Scenario generation for emissions, concentrations, and temperature paths.

- **Handout_DICE.pdf**  
  Summary of DICE equations and calibration strategy.

- **Slides_1A.pdf** and **Slides_1B.pdf**  
  Slides covering IAMs, transition dynamics, policy experiments, and numerical methods.

### Learning objectives

- Understand the structure of IAMs (economy, carbon cycle, climate, damages, abatement).  
- Simulate the DICE model and adjust policy variables.  
- Produce emissions and temperature scenarios.  
- Analyze climate policy from a macroeconomic perspective.

---

## Running the notebooks

### Requirements

Install Python packages:

```
pip install numpy pandas matplotlib scipy
```

### Launching

From the repository root:

```
jupyter notebook
```

Open any `.ipynb` file to begin.

---

## Course outline

### Lecture 1 (4 hours)

1. Climate science basics  
2. Greenhouse gases and radiative forcing  
3. Carbon cycle and temperature models  
4. Growth theory and externalities  
5. Python session: simulate historical temperature paths  

### Lecture 2 (4 hours)

1. Architecture of Integrated Assessment Models  
2. DICE model equations and calibration  
3. Transition dynamics and climate policy  
4. Scenario analysis  
5. Python session: simulate and modify DICE  

---

## License

Materials are free to use for academic teaching and research.

---

For questions or suggestions, contact: **gauthier@vermandel.fr**
