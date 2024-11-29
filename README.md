# Gender-Equality-ML-Project

- **Project Objective**:  
  - Analyze and predict gender pay gaps across industries, job roles, and regions.  
  - Utilize machine learning to identify key factors influencing salary disparities.  
  - Address social, economic, and organizational goals to promote workplace equity and informed decision-making.  

- **Dataset Details**:  
  - Synthetic dataset with the following features:  
    - **Gender**: Male/Female.  
    - **Industry**: Sectors like Technology, Finance, Healthcare, etc.  
    - **Job Role**: Specific roles such as Data Scientist, Financial Analyst.  
    - **Region**: Geographical areas (e.g., North America, Europe).  
    - **Years of Experience**: Total experience in the job role.  
    - **Education Level**: Highest qualification (Bachelor, Master, PhD).  
    - **Salary**: Annual income  (target variable).  

- **Methodology**:  
  - **Data Exploration**: Used descriptive statistics and visualizations to identify trends and gaps.  
  - **Preprocessing**:  
    - Encoded categorical variables.  
    - Standardized continuous features.  
    - Split data into training (80%) and testing (20%) sets.  
  - **Model Selection**:  
    - XGBoost regressor selected for its efficiency in regression tasks.  
    - Tuned hyperparameters for optimal performance.  
  - **Evaluation**:  
    - Metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).  

- **Key Findings**:  
  - Significant predictors of salary include job role, years of experience, industry, and region.  
  - Gender had a measurable but less pronounced effect on salary disparities.  

- **Future Work**:  
  - Integrate real-world data for more accurate, generalizable insights.  
  - Apply advanced modeling techniques and further tune models for better performance.  
  - Expand the dataset with additional features like company size and job level.
