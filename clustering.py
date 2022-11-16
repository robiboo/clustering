import pandas as pd
from sklearn.metrics import *
from sklearn.cluster import KMeans
from sklearn.pipeline import *
from sklearn.preprocessing import MinMaxScaler

def main():
    data = pd.read_csv("./Pokemon.csv")
    categorical = ['Type 1']
    # data_file = pd.get_dummies(data=data, columns=categorical)
    data_file = data.fillna(0)

    data_file = data_file.drop(columns=['Type 2', 'Generation', 'Legendary'])

    # data_type = pd.get_dummies(data=data['Type 1'], columns=categorical)
    # print(data_type)

    pokemon_type = ['Bug', 'Dark', 'Dragon', 'Electric', 'Fairy', 'Fighting', 'Fire', 'Flying', 'Ghost', \
                    'Grass', 'Ground', 'Ice', 'Normal', 'Poison', 'Psychic', 'Rock', 'Steel', 'Water']

    steps = [
        ('scale', MinMaxScaler()),
        ('cluster', KMeans(n_clusters=0, random_state=0)),

    ]
    pipe = Pipeline(steps)

    pokedex = {}
    range_n_clusters = list(range(2, 15))
    for pt in pokemon_type:
        ptype = data_file[data_file['Type 1'] == pt]
        best_cluster = 0
        best_score = 0
        print(pt)
        print('-----------')
        for n in range_n_clusters:
            if n >= ptype.count()[0]:
                break
            pipe.set_params(cluster__n_clusters = n)
            scaler = MinMaxScaler()
            new_data_set = scaler.fit(ptype.loc[:, ~ptype.columns.isin(['#', 'Name', 'Type 1'])])
            new_data_set = new_data_set.transform(ptype.loc[:, ~ptype.columns.isin(['#', 'Name', 'Type 1'])])
            dt = pipe.fit_predict(new_data_set)

            score = silhouette_score(new_data_set, dt)
            if score > best_score:
                best_score = score
                best_cluster = n
            print(f"{n} clusters: {score}")

        pokedex[pt] = [best_cluster, best_score]
        print(f"best number of clusters: {best_cluster}")
        print(f"best score: {best_score}\n")


    # print(data.to_string(index=False))
    for key, value in pokedex.items():
        print(f'\n{key}')
        print('-----', end='')
        ptype = data_file[data_file['Type 1'] == key].copy()
        clusterer = KMeans(n_clusters=value[0])
        clusterer.fit_predict(ptype.loc[:, ~ptype.columns.isin(['#', 'Name', 'Type 1'])])
        ptype['Cluster ID'] = clusterer.labels_
        print_type = ptype.loc[:, ~ptype.columns.isin(['Type 1','#'])]
        for x in range(value[0]):
            print(f"Cluster ", x)
            print_group = print_type[print_type['Cluster ID'] == x].copy()
            print(print_group.loc[:, ~print_group.columns.isin(['Cluster ID'])].to_string())
            mean_hp = print_group['HP'].mean()
            mean_atk = print_group['Attack'].mean()
            mean_def = print_group['Defense'].mean()
            mean_sp_atk = print_group['Sp. Atk'].mean()
            mean_sp_def = print_group['Sp. Def'].mean()
            mean_sp = print_group['Speed'].mean()
            print(f'Mean HP: {mean_hp}')
            print(f'Mean Attack: {mean_atk}')
            print(f'Mean Defense: {mean_def}')
            print(f'Mean Sp. Atk: {mean_sp_atk}')
            print(f'Mean Sp. Def: {mean_sp_def}')
            print(f'Mean Speed: {mean_sp}\n')
        ptype.drop(columns='Cluster ID')







if __name__ == "__main__":
    main()