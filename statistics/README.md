# 📊 Statistics for Data Science – Detailed Notes

These notes cover **Statistics for Data Science & Interviews**, with complete definitions, formulas, practical examples, and pipelines.  

---

## 1. Introduction to Statistics

- **Statistics**: The study of collecting, analyzing, interpreting, presenting, and organizing data.  
- Two main branches:  
  1. **Descriptive Statistics**  
     - Summarizes data using:  
       - **Measures of Central Tendency**: Mean, Median, Mode  
       - **Measures of Dispersion**: Variance, Standard Deviation, Range  
       - **Visual Tools**: Histograms, Boxplots  
     - Example: Average marks of a class.  

  2. **Inferential Statistics**  
     - Makes predictions about populations based on samples.  
     - Uses **statistical tests** like:  
       - Z-test  
       - T-test  
       - ANOVA  
       - Chi-square test  
     - Concepts: **Hypothesis Testing, P-value, Confidence Intervals**  
     - Example: Predicting election results from an exit poll sample.  

📌 **Takeaway:**  
Descriptive = Summarizing, Inferential = Predicting/Testing.

---

## 2. Data, Population, Sample, and Sampling Techniques

### 📌 Key Definitions
- **Data**: Measurable facts or values. (e.g., ages of students, IQ scores).  
- **Population (N)**: Entire dataset of interest.  
- **Sample (n)**: Subset of population used for analysis.  

### 📊 Sampling Techniques
1. **Simple Random Sampling**  
   - Each member has equal chance.  
   - Example: Picking random names from a list.  

2. **Stratified Sampling**  
   - Divide into groups (strata) → take samples from each.  
   - Example: Split by gender, then sample equally.  

3. **Systematic Sampling**  
   - Select every *nth* element.  
   - Example: Every 7th person entering a mall.  

4. **Convenience Sampling**  
   - Based on availability/expertise.  
   - Example: Asking only data science students in a survey.  

📌 **Real-world usage:** Exit polls, drug testing, product surveys.  

---

## 3. Variables and Measurement Scales

### 🔹 Types of Variables
1. **Quantitative (Numerical)**  
   - Discrete: Whole numbers (e.g., number of children).  
   - Continuous: Range values (e.g., height, weight).  

2. **Qualitative (Categorical)**  
   - Cannot apply arithmetic operations.  
   - Example: Gender, Blood group, T-shirt size.  

### 🔹 Measurement Scales
- **Nominal**: Categories without order (e.g., colors).  
- **Ordinal**: Ordered but no magnitude (e.g., movie ratings).  
- **Interval**: Ordered, equal spacing, no true zero (e.g., temperature °C).  
- **Ratio**: Ordered, equal spacing, true zero (e.g., height, weight).  

📌 **Importance:** Scale type determines which **statistical test** is valid.  

---

## 4. Frequency Distribution, Bar Graphs & Histograms

- **Frequency Distribution**: Table showing counts of values.  
- **Cumulative Frequency**: Running total of frequencies.  

### 📊 Visualizations
- **Bar Graphs** → For discrete variables.  
- **Histograms** → For continuous data.  
- **PDF (Probability Density Function)**: Smooth version of histogram (via kernel density).  

📌 **Key difference**:  
Bar = Discrete, Histogram = Continuous.  

---

## 5. Measures of Central Tendency & Dispersion

### 📌 Central Tendency
1. **Mean**: Average (sensitive to outliers).  
2. **Median**: Middle value (robust to outliers).  
3. **Mode**: Most frequent value (useful for categorical).  

### 📌 Dispersion
- **Variance (σ²)**: Average squared deviation from mean.  
- **Standard Deviation (σ)**: √Variance.  
- **Percentiles/Quartiles**: Divide data into 100 or 4 equal parts.  
- **Interquartile Range (IQR)**: Q3 - Q1 → Used for **outlier detection**.  

📊 **Box Plot** → Visual representation of min, Q1, median, Q3, max, and outliers.  

---

## 6. Distributions & Normal Distribution

- **Distribution**: Pattern of how data values are spread.  
- **Normal (Gaussian) Distribution**:  
  - Bell curve, symmetric around mean.  
  - **Empirical Rule (68–95–99.7):**  
    - 68% within 1 SD, 95% within 2 SD, 99.7% within 3 SD.  
- **Standard Normal Distribution**:  
  - Mean = 0, SD = 1.  
  - Formula:  
    \[
    z = \frac{x - \mu}{\sigma}
    \]  
- **Standardization**: Convert to Z-scores.  
- **Normalization**: Scale between [0,1].  

📌 **Usage:** Image preprocessing, comparing student scores across years.  

---

## 7. Hypothesis Testing, Confidence Intervals & P-Value

### 📌 Hypothesis Testing
- **H0 (Null Hypothesis):** No effect/relationship.  
- **H1 (Alternative Hypothesis):** Effect exists.  
- **P-value:** Probability of observing result under H0.  
  - If **p ≤ α (significance level)** → Reject H0.  

### 📌 Confidence Interval
- Interval = Point Estimate ± Margin of Error.  
- Example: 95% CI = “We are 95% confident that true mean lies in this range.”  

### 📌 Errors
- **Type I error (α):** False positive.  
- **Type II error (β):** False negative.  

### 📌 Common Tests
- **Z-test**: Large sample, population SD known.  
- **T-test**: Small sample, population SD unknown.  
- **Chi-Square**: For categorical data.  

---

## 8. Correlation, Covariance & Non-Parametric Tests

### 🔹 Covariance
- Measures joint variability.  
- Positive → Both increase, Negative → One increases while other decreases.  
- Unbounded & scale-dependent.  

### 🔹 Correlation
- Standardized covariance (–1 to +1).  
- **Pearson Correlation**: Linear relationship.  
- **Spearman Correlation**: Ranked (non-linear).  

### 🔹 Distributions
- **Bernoulli**: One trial, success/failure (coin toss).  
- **Binomial**: Multiple Bernoulli trials.  
- **Pareto**: Heavy-tailed, “80/20 rule”.  

---

## 9. Central Limit Theorem (CLT)

- States: Distribution of **sample means** approaches normal as sample size increases (n ≥ 30).  
- Holds true **regardless of original distribution**.  
- Foundation of inferential statistics.  

---

## 10. Summary

- **Descriptive stats** → Summarize data.  
- **Inferential stats** → Test/predict population from sample.  
- **Hypothesis testing, CI, P-value** → Core interview topics.  
- **Normal distribution & CLT** → Foundations of probability/statistics.  
- **Correlation, covariance, tests** → Tools for real-world data analysis.  


