# Project 2 - Ames Housing Sale Predictions

## Problem Statement
Zillow has recently hired a team of data scientists to improve it's methods of home price prediction. As a part of this team, we have been tasked with creating a machine learning model aimed at predicting the price of a home in Ames, Iowa at sale based on a publically available datset. The Ames dataset contains information from the Ames Assessor’s Office used in computing assessed values for individual residential properties sold in Ames, IA from 2006 to 2010.

This will be a proof-of-conecept model geographically restricted to the city of Ames. Zillow would like to determine if it is worth allocating financial resources to this method of prediction in order to expand it to serve as Zillow's primary prediction engine for home sale prices accross the US.

## Background

The Real Estate market has been looking to improve the ways that it predicts home prices to better help customers shop for the home that meets their needs and is within their budget. In pursuit of this goal, the top digital Real Estate companies on the market have agreed to join a competition focused on adapting machine learning techniques to the housing market. Using the Ames, Iowa housing dataset, each company will submit predictions from their two best models to determine if applying machine learning in this way is a viable strategy and to prove to the public who is the one true Real Estate company to rule them all.

<div align="center">
<img src="https://media.tenor.com/SJQuPQYJoB0AAAAd/one-ring-lotr.gif" width="250"/>
</div>

In light of this, Zillow's goal is to be able to produce higher accuracy predictions that it's competitor (i.e. Trulia, Redfin, Realtor.com, etc.). Pre-submissions from each company are accepted until 11:59pm on 2/16/23. 70% of the Ames testing data will be used to produce a score to judge models for all pre-submissions. Once submissions close, the remaining 30% will be used to determine the final scores.

All submitted models will be compared using the metric of Mean Squared Error (MSE). The lower the MSE score is, the better the model performed.

$$MSE = \frac{1}{n}\sum (y_i - \hat{y}_i)^2$$

To achieve Zillow's goals, our team will create utilize three types of linear regression models based on the Ames Housing Dataset that was provided to all companies for this Kaggle competition. These models will be Ordinary Least Squares, Ridge Regression, and LASSO Regression. The top two predictions of the best performing models will be submitted to Kaggle for review.

Because the purpose of this model is solely based on predicting home sale pricing, performance is prioritized over creating a "white-box" model. Zillow has instructed our team that understanding the internal components of the model is not as important as accurate predictions. Proof-of-concept success will be determined by producing an $R^2$ of above 0.90 on the provided data as well as an MSE below 30000 on Kaggle submission.

---

## Datasets

The entire data documentation provided by the Ames Assessor’s Office can be found here: https://jse.amstat.org/v19n3/decock/DataDocumentation.txt

The data dictionary contains all features, provided and engineered, that were used in the models.
- Dummy variables are grouped by their original variable name and denoted under *'Type'*.
- Ordinal features that were converted to a numeric scale are also noted uner *'Type'*.
- Polynomial Features are grouped by the pair of features that created them.


### Data Dictionary:

|Feature|Type|Dataset|Description|
|---|---|---|---|
|**Id**|*Discrete*|Ames, IA Housing|ID number for each home|
|**PID**|*Nominal*|Ames, IA Housing|Parcel identification number  - can be used with city web site for parcel review|
|**MS SubClass**|*Nominal*|Ames, IA Housing|Identifies the type of dwelling involved in the sale|
|**MS Zoning**|*Nominal (Dummy)*|Ames, IA Housing|Identifies the general zoning classification of the sale|
|**Lot Frontage**|*Continuous*|Ames, IA Housing|Linear feet of street connected to property|
|**Lot Area**|*Continuous*|Ames, IA Housing|Lot size in square feet|
|**Street**|*Nominal*|Ames, IA Housing|Type of road access to property|
|**Alley**|*Nominal*|Ames, IA Housing|Type of alley access to property|
|**Lot Shape**|*Ordinal (Dummy)*|Ames, IA Housing|General shape of property|
|**Land Contour**|*Nominal (Dummy)*|Ames, IA Housing|Flatness of the property|
|**Utilities**|*Ordinal*|Ames, IA Housing|Type of utilities available|
|**Lot Config**|*Nominal (Dummy)*|Ames, IA Housing|Lot configuration|
|**Land Slope**|*Ordinal*|Ames, IA Housing|Slope of property|
|**Neighborhood**|*Nominal (Dummy)*|Ames, IA Housing|Physical locations within Ames city limits|
|**Condition 1**|*Nominal (Dummy)*|Ames, IA Housing|Proximity to various conditions|
|**Condition 2**|*Nominal*|Ames, IA Housing|Proximity to various conditions (if more than one is present)|
|**Bldg Type**|*Nominal (Dummy)*|Ames, IA Housing|Type of dwelling|
|**House Style**|*Nominal (Dummy)*|Ames, IA Housing|Style of dwelling|
|**Overall Qual**|*Ordinal*|Ames, IA Housing|Rates the overall material and finish of the house|
|**Overall Cond**|*Ordinal*|Ames, IA Housing|Rates the overall condition of the house|
|**Year Built**|*Discrete*|Ames, IA Housing|Original construction date|
|**Year Remod/Add**|*Discrete*|Ames, IA Housing|Remodel date (same as construction date if no remodeling or additions)|
|**Roof Style**|*Nominal (Dummy)*|Ames, IA Housing|Type of roof|
|**Roof Matl**|*Nominal*|Ames, IA Housing|Roof material|
|**Exterior 1st**|*Nominal (Dummy)*|Ames, IA Housing|Exterior covering on house|
|**Exterior 2nd**|*Nominal (Dummy)*|Ames, IA Housing|Exterior covering on house (if more than one material)|
|**Mas Vnr Type**|*Nominal (Dummy)*|Ames, IA Housing|Masonry veneer type|
|**Mas Vnr Area**|*Continuous*|Ames, IA Housing|Masonry veneer area in square feet|
|**Exter Qual**|*Ordinal (Numeric)*|Ames, IA Housing|Evaluates the quality of the material on the exterior |
|**Exter Cond**|*Ordinal (Numeric)*|Ames, IA Housing|Evaluates the present condition of the material on the exterior|
|**Foundation**|*Nominal (Dummy)*|Ames, IA Housing|Type of foundation|
|**Bsmt Qual**|*Ordinal (Numeric)*|Ames, IA Housing|Evaluates the height of the basement|
|**Bsmt Cond**|*Ordinal (Numeric)*|Ames, IA Housing|Evaluates the general condition of the basement|
|**Bsmt Exposure**|*Ordinal (Numeric)*|Ames, IA Housing|Refers to walkout or garden level walls|
|**BsmtFin Type 1**|*Ordinal (Numeric)*|Ames, IA Housing|Rating of basement finished area|
|**BsmtFin SF 1**|*Continuous*|Ames, IA Housing|Type 1 finished square feet|
|**BsmtFin Type 2**|*Ordinal (Numeric)*|Ames, IA Housing|Rating of basement finished area (if multiple types)|
|**BsmtFin SF 2**|*Continuous*|Ames, IA Housing|Type 2 finished square feet|
|**Bsmt Unf SF**|*Continuous*|Ames, IA Housing|Unfinished square feet of basement area|
|**Total Bsmt SF**|*Continuous*|Ames, IA Housing|Total square feet of basement area|
|**Heating**|*Nominal*|Ames, IA Housing|Type of heating|
|**Heating QC**|*Ordinal (Numeric)*|Ames, IA Housing|Heating quality and condition|
|**Central Air**|*Nominal (Dummy)*|Ames, IA Housing|Central air conditioning|
|**Electrical**|*Ordinal (Numeric)*|Ames, IA Housing|Electrical system|
|**1st Flr SF**|*Continuous*|Ames, IA Housing|First Floor square feet|
|**2nd Flr SF**|*Continuous*|Ames, IA Housing|Second floor square feet|
|**Low Qual Fin SF**|*Continuous*|Ames, IA Housing|Low quality finished square feet (all floors)|
|**Gr Liv Area**|*Continuous*|Ames, IA Housing|Above grade (ground) living area square feet|
|**Bsmt Full Bath**|*Discrete*|Ames, IA Housing|Basement full bathrooms|
|**Bsmt Half Bath**|*Discrete*|Ames, IA Housing|Basement half bathrooms|
|**Full Bath**|*Discrete*|Ames, IA Housing|Full bathrooms above grade|
|**Half Bath**|*Discrete*|Ames, IA Housing|Half baths above grade|
|**Bedroom AbvGr**|*Discrete*|Ames, IA Housing|Bedrooms above grade (does NOT include basement bedrooms)|
|**Kitchen AbvGr**|*Discrete*|Ames, IA Housing|Kitchens above grade|
|**Kitchen Qual**|*Ordinal (Numeric)*|Ames, IA Housing|Kitchen quality|
|**TotRms AbvGrd**|*Discrete*|Ames, IA Housing|Total rooms above grade (does not include bathrooms)|
|**Functional**|*Ordinal*|Ames, IA Housing|Home functionality (Assume typical unless deductions are warranted)|
|**Fireplaces**|*Discrete*|Ames, IA Housing|Number of fireplaces|
|**Fireplace Qu**|*Ordinal (Numeric)*|Ames, IA Housing|Number of fireplaces|
|**Garage Type**|*Nominal (Dummy)*|Ames, IA Housing|Garage location|
|**Garage Yr Blt**|*Discrete*|Ames, IA Housing|Year garage was built|
|**Garage Finish**|*Ordinal (Numeric)*|Ames, IA Housing|Interior finish of the garage|
|**Garage Cars**|*Discrete*|Ames, IA Housing|Size of garage in car capacity|
|**Garage Area**|*Continuous*|Ames, IA Housing|Size of garage in square feet|
|**Garage Qual**|*Ordinal*|Ames, IA Housing|Garage quality|
|**Garage Cond**|*Ordinal*|Ames, IA Housing|Garage condition|
|**Paved Drive**|*Ordinal (Numeric)*|Ames, IA Housing|Paved driveway|
|**Wood Deck SF**|*Continuous*|Ames, IA Housing|Wood deck area in square feet|
|**Open Porch SF**|*Continuous*|Ames, IA Housing|Open porch area in square feet|
|**Enclosed Porch**|*Continuous*|Ames, IA Housing|Enclosed porch area in square feet|
|**3Ssn Porch**|*Continuous*|Ames, IA Housing|Three season porch area in square feet|
|**Screen Porch**|*Continuous*|Ames, IA Housing|Screen porch area in square feet|
|**Pool Area**|*Continuous*|Ames, IA Housing|Pool area in square feet|
|**Pool QC**|*Ordinal*|Ames, IA Housing|Pool quality|
|**Fence**|*Ordinal (Numeric)*|Ames, IA Housing|Fence quality|
|**Misc Feature**|*Nominal*|Ames, IA Housing|Miscellaneous feature not covered in other categories|
|**Misc Val**|*Continuous*|Ames, IA Housing|$ Value of miscellaneous feature|
|**Mo Sold**|*Discrete*|Ames, IA Housing|Month Sold (MM)|
|**Yr Sold**|*Discrete*|Ames, IA Housing|Year Sold (YYYY)|
|**Sale Type**|*Nominal*|Ames, IA Housing|Type of sale|
|**Sale Condition**|*Nominal*|Ames, IA Housing|Condition of sale|
|**SalePrice**|*Continuous*|Ames, IA Housing|Sale price $$|
|**Total Bath**|*Discrete*|Ames, IA Housing - Engineered|Total Bathrooms|
|**Total Area**|*Continuous*|Ames, IA Housing - Engineered|Basement and above grade living area combined (square feet)|
|**Outside Amenity Area**|*Continuous*|Ames, IA Housing - Engineered|All outside area combined (including all porches)|
|**pf_overall**|*Continuous - Polynomial*|Ames, IA Housing|Polynomial features consisitng of ('Overall Qual','Gr Liv Area')|
|**pf_garage**|*Continuous - Polynomial*|Ames, IA Housing|Polynomial features consisitng of ('Garage Area', 'Garage Cars')|
|**pf_ext**|*Discrete - Polynomial*|Ames, IA Housing|Polynomial features consisitng of ('Exter Qual', 'Exter Cond')|
|**pf_bsmt**|*Continuous - Polynomial*|Ames, IA Housing|Polynomial features consisitng of ('Bsmt Qual','Total Bsmt SF', 'Bsmt Cond', 'BsmtFin SF 1', 'Bsmt Exposure')|
|**pf_fire**|*Discrete - Polynomial*|Ames, IA Housing|Polynomial features consisitng of ('Fireplaces', 'Fireplace Qu')|
|**pf_mas**|*Continuous - Polynomial*|Ames, IA Housing|Polynomial features consisitng of ('Mas Vnr Area', 'Mas_Vnr_Stone')|
|**pf_total_qual**|*Discrete - Polynomial*|Ames, IA Housing|Polynomial features consisitng of ('Overall Qual', 'Exter Qual', 'Kitchen Qual', 'Bsmt Qual', 'Heating QC')|

---

## Data Cleaning Process
##### Raw data summary:
- Train dataset has 2051 entries with 81 features (includes SalePrice)
- Test dataset has 878 entries with 80 features (excludes SalePrice)

##### Features:
- 38 features are numeric
- 42 features are categorical

#### Initial Observations:
Some features stored as numbers should be categories:
- Id: ID number
- PID (Nominal): Parcel identification number  - can be used with city web site for parcel review.
    - More information: https://blog.realmanage.com/en-us/pid-pud-how-impact-community-association
- MS SubClass (Nominal): Identifies the type of dwelling involved in the sale.
- Year Built (Discrete): Original construction date
- Year Remod/Add (Discrete): Remodel date (same as construction date if no remodeling or additions)
    
    
Features with Missing Values **(Bold = high number of NaNs)** *(italics = contains numeric NaNs)*:
- *Lot Frontage (Continuous): Linear feet of street connected to property*
- **Alley (Nominal): Type of alley access to property**
- Mas Vnr Type (Nominal): Masonry veneer type
- *Mas Vnr Area (Continuous): Masonry veneer area in square feet*
- Bsmt Qual (Ordinal): Evaluates the height of the basement
- Bsmt Cond (Ordinal): Evaluates the general condition of the basement
- Bsmt Exposure	(Ordinal): Refers to walkout or garden level walls
- BsmtFin Type 1	(Ordinal): Rating of basement finished area
- BsmtFinType 2	(Ordinal): Rating of basement finished area (if multiple types)
- FireplaceQu (Ordinal): Fireplace quality
- Garage Type (Nominal): Garage location
- *Garage Yr Blt (Discrete): Year garage was built*
- Garage Finish (Ordinal): Interior finish of the garage
- Garage Qual (Ordinal): Garage quality
- Garage Cond (Ordinal): Garage condition
- **Pool QC (Ordinal): Pool quality**
- **Fence (Ordinal): Fence quality**
- **Misc Feature (Nominal): Miscellaneous feature not covered in other categories**
    - Elev - Elevator
    - Gar2 - 2nd Garage (if not described in garage section)
    - Othr - Other
    - Shed - Shed (over 100 SF)
    - TenC - Tennis Court
    - NA - None

A majority of the features with missing values can be imputed as 'NA' for none.

'Lot Frontage' and 'Mas Vnr Area' are the only two continuous variables with NaNs. These can be interpretted as 0s, as they correspond to not havnig any masonry and not having any streetfront.

#### Data Cleaning Summary
- Dropped data with mostly missing values.
  - Data with < 10% entries in datasets was dropped.
- Imputed Missing Values.
  - According to the data documentation, all categorical features with NaNs are actually 'NA', which means 'none'.
  - 'Lot Frontage' and 'Mas Vnr Area' are the only two continuous variables with NaNs. These can be interpretted as 0s, as they correspond to not havnig any masonry and not having any streetfront.
- Replaced 'NA' with 'None' for 'Mas Vnr Type'.
  - Additionally cleaned up 'CBlock' that does not appear in both sets of data.
- Grouped Neighborhoods.
  - Some neighborhoods have few values and do not appear in both data sets. These are grouped to maintain consistency between data sets.
- Created a 'Misc' category for 'MS Zoning'.
  - Some values for 'MS Zoning' have few entries and do not appear in both data sets. These are grouped to maintain consistency between data sets.
- Grouped exterior finishes into 'other' category.
  - Some exterior finishes have few entries and do not appear in both data sets. These are grouped to maintain consistency between data sets.

---

## EDA
### Target Feature - Sale Price
<div align="center">
<img src="images/SalePrice Distribution.png" width="700"/>
</div>

The distribution of the SalePrice is skewed right.

### Correlation With Sale Price
<div align="center">
<img src="images/Numeric Feature Correlation with SalePrice.png" width="800"/>
</div>

The Overall Quality rating is most closely correlated with the Sale Price. However, there are many other features that also exhibit strong correlation.

All features with a correaltion of over 50% with Sale Price were individually plotted to identify outliers. Once outliers were determined, they were dropped and the resulting correlation plots are shown below.

### Features With >0.5 Correlation to Sale Price
<div align="center">
<img src="images/Top 6 Correlation Features.png" width="1000"/>
</div>

<div align="center">
<img src="images/Top 7-11 Correlation Features.png" width="1000"/>
</div>

These features show reasonably clear correlation to Sale price that can be generalized with a linear model. However, due to the variations of spread in the data, new features were engineered from these features to creat higher correlations to Sale Price.

---

## Feature Engineering

#### Dummy Features
To analyze categorical features that do not correlate to any ranking scale (nominal features), dummy features were created. Some features were excluded becasue there was not sufficient diversity amongst its values. The following dummy features were included in the model:
- MS Zoning
- Lot Shape
- Land Contour
- Lot Config
- Neighborhood
- Condition 1
- Bldg Type
- House Style
- Roof Style
- Exterior 1st
- Exterior 2nd
- Mas Vnr Type
- Foundation
- Central Air
- Garage Type

#### Ordinal Features
Some categorical feature, however, do map to a ranking scale (ordinal features). These features were converted to a numeric format on scales ranging from 0 to 6 for analyis. Some features were excluded becasue there was not sufficient diversity amongst its values. The following ordinal features were included in the model:
- Exter Qual
- Exter Cond
- Bsmt Qual
- Bsmt Cond
- Bsmt Exposure
- BsmtFin Type 1
- BsmtFin Type 2
- Heating QC
- Electrical
- Kitchen Qual
- FireplaceQu
- Garage Finish
- Paved Drive
- Fence

#### Engineered Features
##### Total Bathrooms
A new feature called 'Total Bath' was engineered to include all bathrooms. bathrooms were counted as follows:
  - 1 per bathroom
  - 0.5 per half bathroom

##### Total Area
A new feature called 'Total Area' was engineered to account for the total suare footage of the home. This includes basement and above ground area.

##### Total Outside Amenity Area
A new feature called 'Outside Amenity Area' was engineered to account for all outdoor amenity spaces. This included all types of porches as well.

#### Polynomial Features
Polynomial features were created to explore interactions between different features. The follow polynomial features were created:
1. Overall quality vs total square footage
1. Garage area vs how many cars fit within the garage
1. Exterior quality vs exterior condition
1. Basement features - baement quality, total basement area, basement condition, basement finish, basement exposure
1. Number of fireplaces vs fireplace quality
1. Masony area vs if the masonry is stone
1. Year built vs remodeled
1. Full bathrooms vs rooms above grade
1. Total quality - overall quality, exterior quality, kitchen quality, basement quality, heating quality

---

## Model Tuning
Once all data had been cleaned and featured engineered, the data was scaled and then three separate models were created.
- Ordinary Least Squares (OLS)
- Ridge Regression
- LASSO Regression

Each model significantly outperformed the baseline, as should be expected. Both the Ridge and LASSO models utilized 10-fold cross-validation to improve their performance.

The residuals for each model are shown below.

<div align="center">
<img src="images/OLS Residuals.png" width="600"/>
</div>


<div align="center">
<img src="images/Ridge Residuals.png" width="600"/>
</div>


<div align="center">
<img src="images/LASSO Residuals.png" width="600"/>
</div>

All three residual plots demonstrate a random pattern, which supports the assumption of linear model. Additionally, the residual plots demonstrate homoscedasticity as the variance remains consistent across all prediction values.

## Conclusions & Recommendations

### Conclusion
Our team of data scientists analyzed the Ames, IA housing dataset to determine if the data provided meaningful information about the Sale Price of each home.

We initially cleaned the data to account for abnormalizites and problems. We then performed EDA on the dataset to discover meaningful correlations between features. Then we engineered features via several methods including, creating dummies, mapping ordinal categories to numbers, creating new features based upon similar features, and creating polynomial features to determine reactions between features. All features were then sacled to prepare them for modeling and regularization.

All of these engineered features were then tested on three linear regression models:
- Ordinary Least Squares (OLS)
- Ridge Regression (l2 penalty)
- LASSO Regresion (l1 penalty)

From these models, it was determined that the Ridge and LASSO models performed best based on their $R^2$ and MSE scores. Both the Ridge and LASSO models utilized 10-fold cross-validation to improve their performance. The Ordinary Least Squares model outperformed the other two models on the training data, but worse on the test data. This is because the model was overfit. Because the Ridge and LASSO models utilize penalties, they can regularize the data to create more robust models that generalize better to new data.

We have concluded that utilizing any of the three linear regression models with this data set can produce accurate predictions above the $R^2$ of 0.90 and < 30000 MSE threshold that Zillow had provided.

### Recommendations
Based on our achievement of the sucess metrics and conclusions, we recommend that Zillow allocate more funding to further develop this home sale price prediction technology. Resources should be distributed to collecting larger and better data sets as well as continuing to refine and improve the current proof-of-concept models.

We also recommend that if Zillow chooses to utilize models to predict the effect of certain aspects of a home on sale price rather than simply utilizing all information to predict the sale price, that the complexity of the model be reduced through removing variables that exhibit multicollinearity. This would likely reduce the overall performance of the model, but would provide more clarity to how individual features affect sale price.

