{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "b45c932c-386f-4eed-944f-24c381e74516",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "from tensorflow import keras\n",
    "from tensorflow.keras import Sequential\n",
    "from tensorflow.keras.activations import relu, softmax\n",
    "from tensorflow.keras.layers import Dense, Dropout\n",
    "from tensorflow.keras.optimizers import Adam\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "72460d43-654c-43c9-87fb-498c5440e167",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_excel(\"Blast Efficiency Data.xlsx\",\n",
    "                  sheet_name= 'Input')\n",
    "df1 = pd.read_excel(\"Blast Efficiency Data.xlsx\",\n",
    "                  sheet_name= 'Output')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "0a84582d-cbc6-44fb-8a30-21348f718b7f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>H (m)</th>\n",
       "      <th>B/S</th>\n",
       "      <th>SR</th>\n",
       "      <th>IH (degree)</th>\n",
       "      <th>T (m)</th>\n",
       "      <th>EF</th>\n",
       "      <th>K (Kg/m³)</th>\n",
       "      <th>MIC (Kg)</th>\n",
       "      <th>Unnamed: 8</th>\n",
       "      <th>Unnamed: 9</th>\n",
       "      <th>Unnamed: 10</th>\n",
       "      <th>Unnamed: 11</th>\n",
       "      <th>Unnamed: 12</th>\n",
       "      <th>Unnamed: 13</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>10.5</td>\n",
       "      <td>0.818182</td>\n",
       "      <td>2.333333</td>\n",
       "      <td>90</td>\n",
       "      <td>2.6</td>\n",
       "      <td>0.606061</td>\n",
       "      <td>0.346320</td>\n",
       "      <td>138.435291</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>9.0</td>\n",
       "      <td>0.840000</td>\n",
       "      <td>2.142857</td>\n",
       "      <td>87</td>\n",
       "      <td>2.4</td>\n",
       "      <td>0.300926</td>\n",
       "      <td>0.171958</td>\n",
       "      <td>170.153019</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   H (m)       B/S        SR  IH (degree)  T (m)        EF  K (Kg/m³)  \\\n",
       "0   10.5  0.818182  2.333333           90    2.6  0.606061   0.346320   \n",
       "1    9.0  0.840000  2.142857           87    2.4  0.300926   0.171958   \n",
       "\n",
       "     MIC (Kg)  Unnamed: 8  Unnamed: 9  Unnamed: 10  Unnamed: 11  Unnamed: 12  \\\n",
       "0  138.435291         NaN         NaN          NaN          NaN          NaN   \n",
       "1  170.153019         NaN         NaN          NaN          NaN          NaN   \n",
       "\n",
       "  Unnamed: 13  \n",
       "0         NaN  \n",
       "1         NaN  "
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "9a3e214c-92ad-40b5-904d-a384afbaae0f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "H (m)            0\n",
       "B/S              0\n",
       "SR               0\n",
       "IH (degree)      0\n",
       "T (m)            0\n",
       "EF               0\n",
       "K (Kg/m³)        0\n",
       "MIC (Kg)         0\n",
       "Unnamed: 8     407\n",
       "Unnamed: 9     407\n",
       "Unnamed: 10    407\n",
       "Unnamed: 11    407\n",
       "Unnamed: 12    407\n",
       "Unnamed: 13    397\n",
       "dtype: int64"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "8901820d-206a-4120-992a-9535c2d70c7e",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "df.drop(['Unnamed: 8','Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13'  ], axis = 1, inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "9d1b7942-614c-4b29-b7f2-2c242bd03459",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>H (m)</th>\n",
       "      <th>B/S</th>\n",
       "      <th>SR</th>\n",
       "      <th>IH (degree)</th>\n",
       "      <th>T (m)</th>\n",
       "      <th>EF</th>\n",
       "      <th>K (Kg/m³)</th>\n",
       "      <th>MIC (Kg)</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>10.5</td>\n",
       "      <td>0.818182</td>\n",
       "      <td>2.333333</td>\n",
       "      <td>90</td>\n",
       "      <td>2.6</td>\n",
       "      <td>0.606061</td>\n",
       "      <td>0.346320</td>\n",
       "      <td>138.435291</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>9.0</td>\n",
       "      <td>0.840000</td>\n",
       "      <td>2.142857</td>\n",
       "      <td>87</td>\n",
       "      <td>2.4</td>\n",
       "      <td>0.300926</td>\n",
       "      <td>0.171958</td>\n",
       "      <td>170.153019</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   H (m)       B/S        SR  IH (degree)  T (m)        EF  K (Kg/m³)  \\\n",
       "0   10.5  0.818182  2.333333           90    2.6  0.606061   0.346320   \n",
       "1    9.0  0.840000  2.142857           87    2.4  0.300926   0.171958   \n",
       "\n",
       "     MIC (Kg)  \n",
       "0  138.435291  \n",
       "1  170.153019  "
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "9afad5d8-05d7-4307-9367-af5174f80ced",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['H (m)', 'B/S', 'SR', 'IH (degree)', 'T (m)', 'EF', 'K (Kg/m³)',\n",
       "       'MIC (Kg)'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "5186ecd1-2fca-461e-9ef7-e4c7357c34ef",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "BE (%)    0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df1.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "16826ee2-2775-4724-888f-b21d9f9a6894",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['BE (%)'], dtype='object')"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df1.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "317e04ad-f708-40ea-a38a-3e66b58be360",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df[['H (m)', 'B/S', 'SR', 'IH (degree)', 'T (m)', 'EF', 'K (Kg/m³)',\n",
    "       'MIC (Kg)']]\n",
    "y = df1['BE (%)']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 141,
   "id": "54fe729e-5b9d-4abf-b63c-dbdc748dd9fd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "H (m)         -0.553843\n",
       "B/S           -0.078492\n",
       "SR            -0.403030\n",
       "IH (degree)   -0.146818\n",
       "T (m)         -0.512317\n",
       "EF             0.278249\n",
       "K (Kg/m³)      0.278249\n",
       "MIC (Kg)       0.034984\n",
       "dtype: float64"
      ]
     },
     "execution_count": 141,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "corr = df.corrwith(df1['BE (%)'])\n",
    "corr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 142,
   "id": "14e921d5-ae79-40ac-855d-de0f594c1d49",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df[['H (m)',  'SR',  'T (m)', 'EF', 'K (Kg/m³)',\n",
    "       ]]\n",
    "y = df1['BE (%)']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 143,
   "id": "f0c6c865-198c-4268-9f15-5a8574f8ee5e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Actual</th>\n",
       "      <th>Predicted</th>\n",
       "      <th>Diff</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>71.7</td>\n",
       "      <td>71.795007</td>\n",
       "      <td>-0.095007</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>67.6</td>\n",
       "      <td>70.274411</td>\n",
       "      <td>-2.674411</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>70.1</td>\n",
       "      <td>71.795007</td>\n",
       "      <td>-1.695007</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>71.5</td>\n",
       "      <td>68.846799</td>\n",
       "      <td>2.653201</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>62.2</td>\n",
       "      <td>66.532216</td>\n",
       "      <td>-4.332216</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>67.5</td>\n",
       "      <td>68.350689</td>\n",
       "      <td>-0.850689</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>67.9</td>\n",
       "      <td>66.909743</td>\n",
       "      <td>0.990257</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>69.8</td>\n",
       "      <td>66.651869</td>\n",
       "      <td>3.148131</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>70.7</td>\n",
       "      <td>70.937310</td>\n",
       "      <td>-0.237310</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>72.6</td>\n",
       "      <td>72.270569</td>\n",
       "      <td>0.329431</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>67.1</td>\n",
       "      <td>67.085235</td>\n",
       "      <td>0.014765</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    Actual  Predicted      Diff\n",
       "0     71.7  71.795007 -0.095007\n",
       "1     67.6  70.274411 -2.674411\n",
       "2     70.1  71.795007 -1.695007\n",
       "3     71.5  68.846799  2.653201\n",
       "4     62.2  66.532216 -4.332216\n",
       "5     67.5  68.350689 -0.850689\n",
       "6     67.9  66.909743  0.990257\n",
       "7     69.8  66.651869  3.148131\n",
       "8     70.7  70.937310 -0.237310\n",
       "9     72.6  72.270569  0.329431\n",
       "10    67.1  67.085235  0.014765"
      ]
     },
     "execution_count": 143,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.metrics import mean_squared_error\n",
    "\n",
    "# Assuming X_train and y_train are your original datasets\n",
    "\n",
    "# Split the data into training and validation sets\n",
    "X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)\n",
    "\n",
    "# Normalize the input data\n",
    "scaler = StandardScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "X_val_scaled = scaler.transform(X_val)\n",
    "\n",
    "# Create and train a Linear Regression model\n",
    "lr = LinearRegression()\n",
    "lr.fit(X_train_scaled, y_train)\n",
    "\n",
    "# Make predictions on the validation set\n",
    "y_pred = lr.predict(X_val_scaled)\n",
    "\n",
    "# Convert y_val and y_pred to numpy arrays\n",
    "y_val_np = np.array(y_val)\n",
    "y_pred_np = np.array(y_pred)\n",
    "\n",
    "# Create a DataFrame for easier comparison\n",
    "pred_df = pd.DataFrame({'Actual': y_val_np.flatten(), 'Predicted': y_pred_np.flatten(), 'Diff': y_val_np.flatten() - y_pred_np.flatten()})\n",
    "pred_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 144,
   "id": "3b13f6c5-d001-4de0-80a3-26d165b97198",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean Squared Error (MSE): 4.329289319625175\n",
      "Mean Absolute Error (MAE): 1.5473111550222298\n",
      "R-squared (R2): 0.4494661093044324\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n",
    "\n",
    "# Calculate Mean Squared Error (MSE)\n",
    "mse = mean_squared_error(y_val, y_pred)\n",
    "\n",
    "# Calculate Mean Absolute Error (MAE)\n",
    "mae = mean_absolute_error(y_val, y_pred)\n",
    "\n",
    "# Calculate R-squared (R2)\n",
    "r2 = r2_score(y_val, y_pred)\n",
    "\n",
    "print(\"Mean Squared Error (MSE):\", mse)\n",
    "print(\"Mean Absolute Error (MAE):\", mae)\n",
    "print(\"R-squared (R2):\", r2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 147,
   "id": "0c812d15-0bcc-48bd-a7b7-597fc6c3964d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA04AAAIhCAYAAAB5deq6AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAABxcklEQVR4nO3dd3RUBd7G8edmUglJIH0CCb2FEEAiEFAUAUECC9grIDaKlWVVXFd0XcXXXV11lyJKEVGxBBAJIqgEFUKTktCLQAKZ0EmD9Hn/YBmNQAomuSnfzzlzdufWZ4JnyMO98xvDbrfbBQAAAAC4LCezAwAAAABAdUdxAgAAAIBSUJwAAAAAoBQUJwAAAAAoBcUJAAAAAEpBcQIAAACAUlCcAAAAAKAUFCcAAAAAKAXFCQAAAABKQXECgBpizpw5MgzD8XB2dlbjxo11//3368iRIxV6rry8PI0ePVpWq1UWi0WdOnWq0OPXNS+++KIMw5CTk5N++eWXi9ZnZ2fL29tbhmFo5MiRV3SOV199VYsWLSrXPhf+mzp48OAVnRMA6hKKEwDUMLNnz1ZCQoJWrFihhx56SJ988omuvfZaZWdnV9g5pk2bpnfffVd//etf9dNPP+nDDz+ssGPXZfXr19fs2bMvWv75558rPz9fLi4uV3zsKylOMTExSkhIkNVqveLzAkBdQXECgBomIiJC3bt3V+/evTVp0iQ9/fTTOnDgQLl/ab6Us2fPSpK2bdsmDw8PPfroo4qOjlaHDh3+8LHPnTv3h49R091xxx364IMPVFRUVGz5zJkzNWzYMLm6ulZJjnPnzslutysgIEDdu3eXm5tblZwXAGoyihMA1HDdu3eXJB06dEiSZLfbNXXqVHXq1EkeHh5q2LChbr311otuEbv++usVERGhH374QT169FC9evU0atQoGYah999/X+fOnXPcFjhnzhxJUk5OjiZOnKhmzZrJ1dVVjRo10rhx43TmzJlix27atKkGDRqkBQsWqHPnznJ3d9dLL72k+Ph4GYahjz/+WM8884ysVqvq16+vwYMH6+jRo8rMzNTDDz8sf39/+fv76/7771dWVlaxY0+ZMkW9evVSYGCgPD091aFDB73++uvKz8+/5OvbsGGDrr32WtWrV0/NmzfXa6+9dlFxOXPmjP785z+refPmcnNzU2BgoAYOHKhdu3Y5tsnLy9M//vEPtW3bVm5ubgoICND999+v48ePl/nPatSoUUpJSdGKFSscy/bs2aOffvpJo0aNuuQ+GRkZmjBhQrGf+ZNPPlnsCqNhGMrOztYHH3zg+DO7/vrrJf16O97y5cs1atQoBQQEqF69esrNzb3srXrLli1Tnz595OPjo3r16qldu3aaPHmyY/0vv/yiO++8UyEhIXJzc1NQUJD69OmjLVu2lPlnAQA1jbPZAQAAf8y+ffskSQEBAZKkRx55RHPmzNHjjz+u//u//9OpU6f097//XT169NDWrVsVFBTk2Ndms+nee+/V008/rVdffVVOTk568skn9fLLL2vlypX6/vvvJUktWrSQ3W7X0KFD9d1332nixIm69tprlZiYqEmTJikhIUEJCQnFrlxs2rRJO3fu1PPPP69mzZrJ09PT8cv+c889p969e2vOnDk6ePCgJkyYoLvuukvOzs7q2LGjPvnkE23evFnPPfecvLy89M477ziOu3//ft19992OIrF161a98sor2rVrl2bNmlXsZ5OWlqZ77rlHf/7znzVp0iQtXLhQEydOVEhIiIYPHy5JyszM1DXXXKODBw/qmWeeUbdu3ZSVlaUffvhBNptNbdu2VVFRkYYMGaIff/xRTz/9tHr06KFDhw5p0qRJuv7667Vx40Z5eHiU+mfVqlUrXXvttZo1a5b69+8vSZo1a5aaNm2qPn36XLT92bNndd111+nw4cN67rnnFBkZqe3bt+uFF15QUlKSvv32WxmGoYSEBN1www3q3bu3/va3v0mSvL29ix1r1KhRiomJ0Ycffqjs7OzL3hY4c+ZMPfTQQ7ruuus0ffp0BQYGas+ePdq2bZtjm4EDB6qwsFCvv/66wsLCdOLECa1Zs+aiAg0AtYodAFAjzJ492y7JvnbtWnt+fr49MzPTvmTJEntAQIDdy8vLnpaWZk9ISLBLsr/xxhvF9k1JSbF7eHjYn376acey6667zi7J/t133110rhEjRtg9PT2LLVu2bJldkv31118vtvzTTz+1S7LPmDHDsaxJkyZ2i8Vi3717d7FtV65caZdkHzx4cLHlTz75pF2S/fHHHy+2fOjQoXZfX9/L/kwKCwvt+fn59rlz59otFov91KlTF72+devWFdsnPDzc3r9/f8fzv//973ZJ9hUrVlz2PJ988oldkj02NrbY8g0bNtgl2adOnXrZfe12u33SpEl2Sfbjx4/bZ8+ebXdzc7OfPHnSXlBQYLdarfYXX3zRbrfb7Z6envYRI0Y49ps8ebLdycnJvmHDhmLH++KLL+yS7EuXLnUs+/2+F1z472b48OGXXXfgwAG73W63Z2Zm2r29ve3XXHONvaio6JKv5cSJE3ZJ9rfeeqvE1wwAtQ236gFADdO9e3e5uLjIy8tLgwYNUnBwsL7++msFBQVpyZIlMgxD9957rwoKChyP4OBgdezYUfHx8cWO1bBhQ91www1lOu+Fq0+/n/p22223ydPTU999912x5ZGRkWrduvUljzVo0KBiz9u1ayfp/LCC3y8/depUsdv1Nm/erD/96U/y8/OTxWKRi4uLhg8frsLCQu3Zs6fY/sHBweratetFuS7c1ihJX3/9tVq3bq2+ffte7qVryZIlatCggQYPHlzs59qpUycFBwdf9HMtyW233SZXV1d99NFHWrp0qdLS0i47SW/JkiWKiIhQp06dip23f//+MgyjXOe95ZZbSt1mzZo1ysjI0NixY2UYxiW38fX1VYsWLfTPf/5Tb775pjZv3nzRrY8AUBtxqx4A1DBz585Vu3bt5OzsrKCgoGIT0Y4ePSq73V7sdrzfat68ebHn5ZmmdvLkSTk7OztuCbzAMAwFBwfr5MmTZT62r69vsecXhiJcbnlOTo7q16+v5ORkXXvttWrTpo3efvttNW3aVO7u7lq/fr3GjRt30QAKPz+/i87t5uZWbLvjx48rLCzsslml8z/XM2fOXHZ4w4kTJ0rc/7c8PT11xx13aNasWWrSpIn69u2rJk2aXPa8+/btu+xtdeU5b1n+rC98Xqtx48aX3cYwDH333Xf6+9//rtdff11//vOf5evrq3vuuUevvPKKvLy8ypwJAGoSihMA1DDt2rVTVFTUJdf5+/vLMAz9+OOPl5yU9vtll7uqcCl+fn4qKCjQ8ePHi5Unu92utLQ0XX311Vd87LJatGiRsrOztWDBgmJl448MJQgICNDhw4dL3Mbf319+fn5atmzZJdeXtyyMGjVK77//vhITE/XRRx+VeF4PD4+LPrv12/VlVZY/jwt/rqX9PJo0aaKZM2dKOj/c4rPPPtOLL76ovLw8TZ8+vcyZAKAmoTgBQC0yaNAgvfbaazpy5Ihuv/32Cj12nz599Prrr2vevHl66qmnHMtjY2OVnZ19yeEGFe3CL/+/LYB2u13vvffeFR/zpptu0gsvvKDvv//+srctDho0SPPnz1dhYaG6det2xee6IDo6WqNGjVJ6erqGDRt22e0GDRqkV199VX5+fmrWrFmJx/z9lbQr0aNHD/n4+Gj69Om68847y1S2Wrdureeff16xsbHatGnTHzo/AFRnFCcAqEV69uyphx9+WPfff782btyoXr16ydPTUzabTT/99JM6dOigMWPGXNGx+/Xrp/79++uZZ55RRkaGevbs6Ziq17lzZ913330V/GouncHV1VV33XWXnn76aeXk5GjatGk6ffr0FR/zySef1KeffqohQ4bo2WefVdeuXXXu3DmtWrVKgwYNUu/evXXnnXfqo48+0sCBA/XEE0+oa9eucnFx0eHDh7Vy5UoNGTKkxAJ0KReu2JSWLTY2Vr169dJTTz2lyMhIFRUVKTk5WcuXL9ef//xnR5Hr0KGD4uPj9dVXX8lqtcrLy0tt2rQpV6b69evrjTfe0IMPPqi+ffvqoYceUlBQkPbt26etW7fqv//9rxITE/Xoo4/qtttuU6tWreTq6qrvv/9eiYmJevbZZ8t1PgCoSShOAFDLvPvuu+revbveffddTZ06VUVFRQoJCVHPnj0vGpRQHoZhaNGiRXrxxRc1e/ZsvfLKK/L399d9992nV199tUq+RLVt27aKjY3V888/r5tvvll+fn66++67NX78eN10001XdEwvLy/99NNPevHFFzVjxgy99NJLatiwoa6++mo9/PDDkiSLxaLFixfr7bff1ocffqjJkyfL2dlZjRs31nXXXVchXxB8KZ6envrxxx/12muvacaMGTpw4IA8PDwUFhamvn37qmnTpo5t3377bY0bN0533nmnY4x5eYZHXPDAAw8oJCRE//d//6cHH3xQdrtdTZs21YgRIySdH7jRokULTZ06VSkpKTIMQ82bN9cbb7yhxx57rIJeOQBUP4bdbrebHQIAAAAAqjPGkQMAAABAKShOAAAAAFAKihMAAAAAlILiBAAAAACloDgBAAAAQCmqTXGaPHmyDMPQk08+edlt4uPjZRjGRY9du3ZVXVAAAAAAdU61+B6nDRs2aMaMGYqMjCzT9rt375a3t7fjeUBAQJnPVVRUpNTUVHl5eZXpG9EBAAAA1E52u12ZmZkKCQmRk1PJ15RML05ZWVm655579N577+kf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M2eqoKCg2D9C2e12ubi46PTp02rYsKFjeWhoKO/BAGodT09Px3vbO++8o969e+ull17SpEmTtGXLFsd2vr6+kqTZs2fr8ccf17Jly/Tpp5/q+eef14oVK9S9e/cync/Jyemi26Hz8/Md/z87O1s33nijbrzxRs2bN08BAQFKTk5W//79lZeXd1H2Cy6Uq7i4uIsuLLi5uUlShd6GzWecKsmAAQOUl5envLw89e/fv0z7hIeHKzs7u5KTAUDtVtL7b0FBgebOnas33nhDW7ZscTy2bt2qJk2a6KOPPjIpNQCYZ9KkSfrXv/6lY8eOqWXLlo7HheIkSZ07d9bEiRO1Zs0aRURE6OOPP5YktWvX7qKvafj984CAANlsNsfzwsJCbdu2zfF8165dOnHihF577TVde+21atu2bbHBEJcTHh4uNzc3JScnF8vdsmVLhYaGOrYpLV9ZccWpklgsFu3cudPx/3/r5MmTuu222zRq1ChFRkbKy8tLGzdu1Ouvv64hQ4YU2zYzM1NpaWnFltWrV0/e3t6V+wIAoIYq6f13yZIlOn36tB544AH5+PgUW3frrbdq5syZjqtV0vn369+/Bzdo0EDu7u6VlB4Aqt7111+v9u3b69VXX9V///vfYusOHDigGTNm6E9/+pNCQkK0e/du7dmzR8OHD5ckPfHEExoxYoSioqJ0zTXX6KOPPtL27dvVvHlzxzFuuOEGjR8/XnFxcWrRooX+/e9/F/vy2bCwMLm6uuo///mPRo8erW3btunll18uNbeXl5cmTJigp556SkVFRbrmmmuUkZGhNWvWqH79+hoxYoRGjx6tN954Q+PHj9cjjzyin3/++bKTVEvDFadK5O3tfcmCU79+fXXr1k3//ve/1atXL0VEROhvf/ubHnrooYv+Y33hhRdktVqLPZ5++umqegkAUCNd7v135syZ6tu370WlSTr/2dMtW7Zo06ZNjmV9+/a96D140aJFlRkdAEwxfvx4vffee0pJSSm2vF69etq1a5duueUWtW7dWg8//LAeffRRPfLII5KkO+64Qy+88IKeeeYZdenSRYcOHdKYMWOKHWPUqFEaMWKEhg8fruuuu07NmjVT7969HesDAgI0Z84cff755woPD9drr72mf/3rX2XK/fLLL+uFF17Q5MmT1a5dO/Xv319fffWVmjVrJul8KYuNjdVXX32ljh07avr06Xr11Vev6Gdk2Cvyxj8AAAAAddqLL76oRYsWFfu8VG3AFScAAAAAKAXFCQAAAABKwa16AAAAAFAKrjgBAAAAQCkoTgAAAABQCooTAAAAAJSC4gQAAAAApaA4AQAAAEApKE4AAFyCYRhatGiR2TEAANUExQkAUG2NHDlShmFo9OjRF60bO3asDMPQyJEjy3Ss+Ph4GYahM2fOlGl7m82mm266qRxpAQC1GcUJAFCthYaGav78+Tp37pxjWU5Ojj755BOFhYVV+Pny8vIkScHBwXJzc6vw4wMAaiaKEwCgWrvqqqsUFhamBQsWOJYtWLBAoaGh6ty5s2OZ3W7X66+/rubNm8vDw0MdO3bUF198IUk6ePCgevfuLUlq2LBhsStV119/vR599FGNHz9e/v7+6tevn6SLb9U7fPiw7rzzTvn6+srT01NRUVFat26dJGnr1q3q3bu3vLy85O3trS5dumjjxo2V+WMBAFQxZ7MDAABQmvvvv1+zZ8/WPffcI0maNWuWRo0apfj4eMc2zz//vBYsWKBp06apVatW+uGHH3TvvfcqICBA11xzjWJjY3XLLbdo9+7d8vb2loeHh2PfDz74QGPGjNHq1atlt9svOn9WVpauu+46NWrUSIsXL1ZwcLA2bdqkoqIiSdI999yjzp07a9q0abJYLNqyZYtcXFwq94cCAKhSFCcAQLV33333aeLEiTp48KAMw9Dq1as1f/58R3HKzs7Wm2++qe+//17R0dGSpObNm+unn37Su+++q+uuu06+vr6SpMDAQDVo0KDY8Vu2bKnXX3/9suf/+OOPdfz4cW3YsMFxnJYtWzrWJycn6y9/+Yvatm0rSWrVqlVFvXQAQDVBcQIAVHv+/v6KiYnRBx98ILvdrpiYGPn7+zvW79ixQzk5OY7b7C7Iy8srdjvf5URFRZW4fsuWLercubOjNP3e+PHj9eCDD+rDDz9U3759ddttt6lFixZleGUAgJqC4gQAqBFGjRqlRx99VJI0ZcqUYusu3DIXFxenRo0aFVtXlgEPnp6eJa7/7W19l/Liiy/q7rvvVlxcnL7++mtNmjRJ8+fP17Bhw0o9NwCgZmA4BACgRhgwYIDy8vKUl5en/v37F1sXHh4uNzc3JScnq2XLlsUeoaGhkiRXV1dJUmFhYbnPHRkZqS1btujUqVOX3aZ169Z66qmntHz5ct18882aPXt2uc8DAKi+KE4AgBrBYrFo586d2rlzpywWS7F1Xl5emjBhgp566il98MEH2r9/vzZv3qwpU6bogw8+kCQ1adJEhmFoyZIlOn78uLKyssp87rvuukvBwcEaOnSoVq9erV9++UWxsbFKSEjQuXPn9Oijjyo+Pl6HDh3S6tWrtWHDBrVr165CXz8AwFwUJwBAjeHt7S1vb+9Lrnv55Zf1wgsvaPLkyWrXrp369++vr776Ss2aNZMkNWrUSC+99JKeffZZBQUFOW77KwtXV1ctX75cgYGBGjhwoDp06KDXXntNFotFFotFJ0+e1PDhw9W6dWvdfvvtuummm/TSSy9VyGsGAFQPhv1Sc1cBAAAAAA5ccQIAAACAUlCcAAAAAKAUFCcAAAAAKAXFCQAAAABKQXECAAAAgFJQnAAAAACgFBQnAAAAACgFxQkAAAAASkFxAgAAAIBSUJwAAAAAoBQUJwAAAAAoxf8DJ4MD/HhpW4QAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 1000x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "\n",
    "\n",
    "# Plot the metrics\n",
    "metrics = ['MSE', 'MAE', 'R-squared']\n",
    "values = [mse, mae, r2]\n",
    "\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.plot(metrics, values, marker='o')  # Use plt.plot() instead of plt.line()\n",
    "plt.xlabel('Metrics')\n",
    "plt.ylabel('Value')\n",
    "plt.title('Performance Metrics')\n",
    "plt.grid(False)  # Add grid lines for better readability\n",
    "plt.show()\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5d40f266-0cc9-4084-a5e0-c021338e0259",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
