#!/usr/bin/env python
# coding: utf-8

# ## ALGORITMOS DE APRENDIZAJE NO SUPERVISADO
# ## ALEXANDER PARRA MOLANO

# In[2]:


# Importar las librerías necesarias
import pandas as pd

# Cargar el dataset Mall_Customers desde un archivo CSV (cambia la ruta si es necesario)
ruta_archivo = 'C:/Users/aparram/Videos/DataSet/Mall_Customers.csv'  # Cambia esta ruta a la ubicación de tu archivo
data = pd.read_csv(ruta_archivo)

# Ver las primeras filas del dataset
data.head()


# In[4]:


# Para la manipulación de datos
import pandas as pd
import numpy as np
# Para la visualización
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


# Para dividir el dataset y construir el modelo de Árbol de Decisión
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# In[8]:


# Para dividir el dataset y construir el modelo de Árbol de Decisión
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# In[10]:


# Resumen estadístico de las variables numéricas
data.describe()


# In[14]:


# Visualización de las distribuciones de las variables numéricas
data.hist(bins=15, figsize=(15, 10), color='skyblue', edgecolor='black')
plt.suptitle("Distribución de Variables Numéricas")
plt.show()


# In[16]:


# Gráfico de barras para visualizar la distribución de género
sns.countplot(x='Gender', data=data, palette='viridis')
plt.title("Distribución de Clientes por Género")
plt.show()


# In[18]:


# Gasto Anual vs Edad
sns.scatterplot(x='Age', y='Annual Income (k$)', data=data, hue='Gender', palette='coolwarm')
plt.title("Gasto Anual vs Edad")
plt.show()

# Gasto Anual vs Ingreso Anual
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=data, hue='Gender', palette='coolwarm')
plt.title("Gasto Anual vs Ingreso Anual")
plt.show()


# In[22]:


# Convertir columnas categóricas en variables numéricas con One-Hot Encoding
data_encoded = pd.get_dummies(data, drop_first=True)

# Matriz de correlación con datos numéricos
correlation_matrix = data_encoded.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Matriz de Correlación entre Variables")
plt.show()


# In[24]:


# Verificar valores faltantes en cada columna
missing_values = data.isnull().sum()
print("Valores faltantes por columna:")
print(missing_values)


# In[28]:


# Convertir columna 'Gender' a valores numéricos
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})

# Confirmar si la conversión fue exitosa
print("Valores únicos en la columna 'Gender':", data['Gender'].unique())


# In[30]:


# Revisar nuevamente si existen valores nulos en el dataset
print("Valores nulos por columna después del preprocesamiento:")
print(data.isnull().sum())


# In[32]:


# Histogramas para visualizar la distribución de las variables numéricas
data.hist(bins=20, figsize=(15, 10), color='skyblue', edgecolor='black')
plt.suptitle("Distribución de Variables Numéricas", fontsize=16)
plt.show()


# In[34]:


# Diagramas de dispersión entre variables relevantes
plt.figure(figsize=(15, 10))

# Edad vs Ingreso Anual
plt.subplot(2, 2, 1)
sns.scatterplot(data=data, x="Age", y="Annual Income (k$)", hue="Gender", palette="Set2")
plt.title("Edad vs Ingreso Anual")

# Ingreso Anual vs Puntuación de Gasto
plt.subplot(2, 2, 2)
sns.scatterplot(data=data, x="Annual Income (k$)", y="Spending Score (1-100)", hue="Gender", palette="Set2")
plt.title("Ingreso Anual vs Puntuación de Gasto")

# Edad vs Puntuación de Gasto
plt.subplot(2, 2, 3)
sns.scatterplot(data=data, x="Age", y="Spending Score (1-100)", hue="Gender", palette="Set2")
plt.title("Edad vs Puntuación de Gasto")

plt.tight_layout()
plt.show()


# In[36]:


# Matriz de correlación
correlation_matrix = data.corr()

# Visualización de la matriz de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Matriz de Correlación")
plt.show()


# In[38]:


# Histogramas para analizar la distribución de las variables numéricas
data.hist(figsize=(15, 10), bins=20, color='teal', edgecolor='black')
plt.suptitle("Distribución de Variables Numéricas", fontsize=16)
plt.show()


# In[40]:


# Diagramas de dispersión entre variables importantes
plt.figure(figsize=(15, 10))

# Ingresos Anuales vs Puntuación de Gastos
plt.subplot(2, 2, 1)
sns.scatterplot(data=data, x="Annual Income (k$)", y="Spending Score (1-100)", hue="Gender", palette="Set2")
plt.title("Ingresos Anuales vs Puntuación de Gastos")
plt.xlabel("Ingresos Anuales (k$)")
plt.ylabel("Puntuación de Gastos")

# Edad vs Puntuación de Gastos
plt.subplot(2, 2, 2)
sns.scatterplot(data=data, x="Age", y="Spending Score (1-100)", hue="Gender", palette="Set2")
plt.title("Edad vs Puntuación de Gastos")
plt.xlabel("Edad")
plt.ylabel("Puntuación de Gastos")

# Edad vs Ingresos Anuales
plt.subplot(2, 2, 3)
sns.scatterplot(data=data, x="Age", y="Annual Income (k$)", hue="Gender", palette="Set2")
plt.title("Edad vs Ingresos Anuales")
plt.xlabel("Edad")
plt.ylabel("Ingresos Anuales (k$)")

plt.tight_layout()
plt.show()


# In[44]:


print(data.columns)


# In[46]:


print(data[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].isnull().sum())


# In[48]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Escalado de las variables numéricas
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[["Age", "Annual Income (k$)", "Spending Score (1-100)"]])

# Verificamos el escalado
print("Escalado completado. Primeras filas de los datos escalados:")
print(data_scaled[:5])

# Método del codo
inertia = []
range_clusters = range(1, 11)

for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)

# Graficar el método del codo
plt.figure(figsize=(8, 5))
plt.plot(range_clusters, inertia, marker='o', linestyle='--')
plt.xlabel('Número de clústeres')
plt.ylabel('Inercia')
plt.title('Método del codo para determinar el número óptimo de clústeres')
plt.show()


# In[50]:


# Entrenar el modelo K-Means con el número óptimo de clústeres
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(data_scaled)

# Agregar los clústeres al DataFrame original
data["Cluster"] = kmeans.labels_

# Mostrar los primeros registros con el clúster asignado
print("Primeros registros con clústeres asignados:")
print(data.head())


# In[52]:


from sklearn.decomposition import PCA

# Reducir dimensiones a 2 componentes principales
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

# Crear un DataFrame con las componentes principales y los clústeres
pca_df = pd.DataFrame(data_pca, columns=["PCA1", "PCA2"])
pca_df["Cluster"] = data["Cluster"]

# Visualización de los clústeres
plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=pca_df, x="PCA1", y="PCA2", hue="Cluster", palette="Set1", s=100
)
plt.title("Visualización de Clústeres en el Espacio PCA")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend(title="Clúster")
plt.show()


# In[56]:


# Importancia de las características basada en la varianza explicada por PCA
pca_components = pd.DataFrame(pca.components_, columns=["Age", "Annual Income (k$)", "Spending Score (1-100)"], index=["PCA1", "PCA2"])
print("Contribución de las características por cada componente principal:")
print(pca_components.T)


# In[58]:


# Entrenamiento del modelo con hiperparámetros ajustados
kmeans_optimized = KMeans(n_clusters=5, init="k-means++", n_init=10, random_state=42)
kmeans_optimized.fit(data_scaled)

# Asignar las etiquetas de los clústeres al dataset original
data["Cluster"] = kmeans_optimized.labels_

# Visualización de las etiquetas generadas
print("Etiquetas generadas por el modelo:")
print(data["Cluster"].value_counts())


# In[60]:


from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Calcular métricas de desempeño
silhouette_avg = silhouette_score(data_scaled, kmeans_optimized.labels_)
calinski_harabasz = calinski_harabasz_score(data_scaled, kmeans_optimized.labels_)

print(f"Coeficiente de Silhouette: {silhouette_avg:.3f}")
print(f"Índice de Calinski-Harabasz: {calinski_harabasz:.3f}")


# In[62]:


plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=data_scaled[:, 1],  # Ingreso anual (normalizado)
    y=data_scaled[:, 2],  # Puntuación de gasto (normalizado)
    hue=kmeans_optimized.labels_,
    palette='viridis',
    s=100
)
plt.title('Visualización de Clústeres')
plt.xlabel('Ingreso Anual (normalizado)')
plt.ylabel('Puntuación de Gasto (normalizado)')
plt.legend(title='Clúster')
plt.grid(True)
plt.show()


# In[64]:


cluster_counts = pd.Series(kmeans_optimized.labels_).value_counts().sort_index()

plt.figure(figsize=(8, 5))
sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='viridis')
plt.title('Distribución de Clientes en Clústeres')
plt.xlabel('Clúster')
plt.ylabel('Cantidad de Clientes')
plt.grid(axis='y')
plt.show()


# In[ ]:




