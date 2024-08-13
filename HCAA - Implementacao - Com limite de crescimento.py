'''
HCAA COM LIMIT ON GROWTH
'''
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as hr
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform, pdist

class Tree:
    '''
    Classe Tree serve para criar a árvore que será usada no estágio adicional e nos ajudará a
    atribuir peso para os clusters e ativos contidos nesses clusters.

    Parameters
    ---------------
    value: int
           É o valor que será guardado por nó, no nosso caso esse valor guardará o indice do cluster criado
           ou o indice dos ativos, que são as folhas.
    weight: float
           É o peso que será atribuido a cada cluster ou ativo
    '''
    def __init__(self, value, weight = None) -> None:
        self.left = None
        self.right = None
        self.data = value
        self.weight = weight

def get_stocks(asset, s_date, e_date):
    '''
    Essa função como o próprio nome já diz tem como objetivo pegar os dados dos ativos que iremos usar
    para performar todo o processo de criação do portfolio.

    Para obter esses dados usamos a API de finanças do Yahoo, yfinance

    Parameters
    ------------
    asset: list
           Lista de strings onde cada string é o ticker do ativo que desejamos obter os dados
    s_date: str
            É a string que usamos para informar a data de inicio para obter os dados, ou seja, queremos
            que os dados obtidos seja a partir dessa data.
    e_date: str
            Semelhante ao s_date, e_date é a data final que usamos para informar até o momento que
            queremos os dados.

    return
    ------------
    data : dataframe pandas
            Dataframe contendo todos os dados dos ativos que passamos como parametro para a API
    '''
    data = yf.download(asset, start = s_date, end = e_date)['Adj Close']
    return data

def get_correlation(data):
    '''
    Obtem a correlação da matriz data, para obter a correlação usamos o metodo  Pearson

    Parameters
    ------------
    data: dataframe pandas
            É a matriz obtida através da API do Yahoo e contem todos os dados dos ativos passados

    Return
    ------------
    data.corr : dataframe pandas
            Matriz contendo a correlação entre os ativos
    '''
    return data.corr(method='pearson')

def calc_distance(correlation):
    '''
    O processo de hierarchical clustering precisa de uma medida de distance, então para isso
    usaremos a medida de Mantegna 1999.

    Di,j = √0.5 * (1 - Pi,j) *P = representação de rho

    Parameters
    ------------
    correlation: dataframe pandas
        Dataframe contendo a correlação entre os ativos

    Return
    ------------
    distance: ndarray
        Matriz contendo a distancia calculada com base na correlação fornecida
    '''
    distance = np.sqrt(0.5 * (1 - correlation))
    return distance

def hierarchical_clustering(euclidean_distance, linkage):
    '''
    Performa o calculo da matriz de linkage usando a biblioteca scipy e o metodo linkage

    Parameters
    ------------
    euclidean_distance: ndarray
        Matriz contendo a distancia euclidiana
    linkage: str
        String para informar qual será o metodo de linkage utilizado

    Return
    ------------
    clustering_matrix: ndarray
        A matriz de hierarchical clustering codificada como uma matriz de link
    '''
    clustering_matrix = hr.linkage(euclidean_distance, method = linkage, optimal_ordering = True)
    return clustering_matrix

def get_idx_cluster_merged(clustering, last_cluster, leaves):
    '''
    Através da matriz de linkage podemos obter os indices dos clusters que foram combinados
    usando para isso a coluna 1 e 2 dessa matriz.

    Essa função retorna um dicionario onde a chave é um indice do cluster criado para comportar
    os dois outros clusters que foram combinados, já os valores são os clusters que foram
    combinados.

    Caso tenha um ponto de cutoff o indice que antes teria como valor dois clusters
    agora será formado portodas as folhas que pertenciam aos clusters anteriores ao indice em
    questão.

    Parameters
    -----------------
    clustering : ndarray
                É a matriz de linkage resultante do processo de hierarchical clustering
    last_cluster: int
                É o valor do ultimo cluster que está contido no ponto de cutoff
    leaves: list
                É a lista contendo as folhas dos clusters que foram removidos pelo ponto de cutoff
                e que sera atribuida ao novo cluster criado contendo todas essas folhas

    return
    -----------------
    cluster_merged: dict
                Dicionario contendo como chave os indices dos clusters que foram criados através
                da combinação de outros, e valores os indices dos clusters que foram envolvidos na
                criação do clusters
    '''
    cluster_merged = {}
    pos = (last_cluster - len(clustering)) - 1
    len_stocks = clustering.shape[0] + 1
    idxs_cla = clustering[pos:, 0].tolist(); idxs_clb = clustering[pos:, 1].tolist()
    for i in range(pos, len(clustering)):
        if i + len_stocks == last_cluster and pos != 0: #if para caso tenha cutoff
            cluster_merged[f"{len_stocks + i}"] = leaves  #atribui as folhas ao indice em questao
        else:
            cluster_merged[f"{len_stocks + i}"] = [int(idxs_cla[i-pos]), int(idxs_clb[i-pos])]
    return cluster_merged

def create_tree(cluster, dicio, raiz, last_cluster):
    '''
    Nessa funcao temos como objetivo pegar o dicionario e usar os dados contidos ali para criar a arvore.
    Para isso, recebemos os valores da ultima chave no parametro 'cluster', com isso verificamos o valor
    de cluster[0] >= 10, se for maior de 10 significa que foi um cluster criado no processo de linkage e
    portanto tem ativos ou clusters associados a ele. O mesmo vale para o cluster[1].

    Usamos cluster[0] para representar os ativos ou clusters que estão contidos na subarvore esquerda, e
    cluster[1] para representar os ativos ou clusters que estão contidos na subarvore direita.

    Tendo essas duas explicações prévias, começamos o codigo de fato.

    Se cluster[0] >= 10 criamos o nó esquerdo da arvore contendo esse valor, pois abaixo dele terá dois ativos
    ou outros clusters. O mesmo vale para o cluster[1].

    Após isso precisamos atualizar os valores de cluster[0] e cluster[1], para fazer isso precisamos buscar
    os valores no dicionario, porém precisamos checar se o cluster ta contido no dicionario e se o mesmo
    é diferente de last_cluster, esse processo é importante para caso tenha sido definido um ponto de cutoff
    os clusters que não fazem parte do dicionario não terão valores para atualizar, isso significa que aquele
    cluster não tem uma subarvore, portanto cluster[0] ou cluster[1] passa a ser o seu proprio valor.

    A função art_leaf_node só sera chamada quando o valor contido em cluster[0] ou cluster[1] for uma lista,
    por isso a verificação se o cluster é diferente de int, pois pode acontecer que cluster seja apenas um
    int em caso de cutoff definido.

    A cada chamada recursiva da função passamos o novo valor de cluster, e também passamos a raiz.left ou
    raiz.right para que nas proximas execuções a arvore continue sendo criada

    Exemplo esquematico sem cutoff                  | Exemplo esquematico com cutoff de 1.0
    ------------------------------------------------|------------------------------------------------
                [14, 17]                            |                   [14, 17]
        [14]                 [17]                   |              [14]          [17]
     [8]    [4]        [16]          [15]           |                       [16]      [15]
                    [3]    [13]   [7]    [12]       |
                        [9]    [5]   [11]    [2]    | *Como explicado nesse caso a função checa_dicio retornaria
                                  [1]    [10]       | falso pois o cluster 14 e 15 nao esta contido no dicionario,
                                      [6]    [0]    | portanto não tem uma subarvore. Já o valor 16 está sim
                                                    | no dicionario, porém devido ao cutoff as folhas não aparecerão
                                                    | somente o cluster 16, que nesse caso seria um cluster novo.
    Parameters
    ----------
    cluster : list
        Lista contendo dois elementos, esses elementos são os indices dos clusters que foram combinados na
        criação do ultimo cluster
    dicio : dict
        Dicionario onde a chave é o indice dos clusters que foram criados através da combinação de outros
        clusters, e os valores são os indices desses clusters
    raiz : Tree
        Arvore que iremos criar com base no dicionario e os clusters que sao seus valores
    last_cluster : int
        Caso tenha um ponto de cutoff esse parametro irá nos dizer qual foi o ultimo cluster criado que
        esta contido no ponto de cutoff

    Return
    ------
    cluster : list
        Lista contendo dois elementos ou apenas um, esses elementos são os indices dos clusters que
        foram combinados na criação do ultimo cluster
    '''
    if type(cluster) == int:
        return cluster
    if cluster[0] < 10 and cluster[1] < 10:
        raiz.left = Tree(cluster[0])
        raiz.right = Tree(cluster[1])
        return cluster
    if cluster[0] >= 10:
        raiz.left = Tree(cluster[0])
        cluster[0] = dicio[f"{int(cluster[0])}"] if checa_dicio(cluster[0], dicio, last_cluster) else cluster[0]
        if type(cluster[0]) != int: atr_leaf_node(cluster[0], raiz.left)
        create_tree(cluster[0], dicio, raiz.left, last_cluster)
    if cluster[1] >= 10:
        raiz.right = Tree(cluster[1])
        cluster[1] = dicio[f"{int(cluster[1])}"] if checa_dicio(cluster[1], dicio, last_cluster) else cluster[1]
        if type(cluster[1]) != int: atr_leaf_node(cluster[1], raiz.right)
        create_tree(cluster[1], dicio, raiz.right, last_cluster)
    return cluster

def checa_dicio(cluster, dicio, last_cluster):
    '''
    Essa função nos ajudará a saber se cluster esta contido no dicionario e se é diferente de last_cluster
    Essa função é importante para caso tenha sido definido um ponto de cutoff, pois caso tenha sido passado
    precisamos barrar a criação da subarvore desse cluster e é por esse motivo que verificamos se
    cluster != last_cluster

    Parameters
    ----------
    cluster: int
        Inteiro que simboliza uma chave para buscar em dicio
    dicio: dict
        Dicionario de cluster
    last_cluster: int
        Inteiro representando o ultimo cluster que esta contido no valor de cutoff

    Return
    ------
    bool
        Boleano para sabermos cluster esta no dicionario e se é diferente de last_cluster
    '''
    if str(cluster) in dicio and cluster != last_cluster:
        return True
    return False

def atr_leaf_node(cluster, raiz):
    '''
    Na função create_tree temos um problema que os valores menores que 10 não seriam colocados na arvore,
    isso acontece devido ao fato de so querermos valores maiores que 10, portanto essa função foi criada
    ela verifica se um dos valores é menor que 10 e se o outro é maior que 10, e guarda o valor menor que
    10 na arvore

    Parameters
    ----------
    cluster : list
        Lista contendo dois elementos, esses elementos são os indices dos clusters que foram combinados na
        criação do ultimo cluster
    raiz : Tree
        Arvore para ser adicionado o valor menor que 10, que nesse caso será uma folha
    '''
    temp = cluster
    if temp[0] <= 10 and temp[1] >= 10:
        raiz.left = Tree(temp[0])
    if temp[0] >= 10 and temp[1] <= 10:
        raiz.right = Tree(temp[1])

def weight_tree(arvore):
    '''
    Essa função é usada para dar peso para os ativos e para os cluster, como um dendrogram é uma arvore binaria
    criamos uma arvore e definimos o peso para os nós.

    Parameters
    ----------
    arvore : Tree
        Raiz da arvore

    Return : tuple
    --------------
    vet_weight : list
        vetor contendo o peso de cada folha da arvore
    tree_leaves : list
        Vetor contendo as folhas da arvore
    '''
    arvore_aux = arvore
    pilha = list()
    weight = 0
    vet_weight = []
    tree_leaves = []
    while ( (arvore_aux != None) or (not pilha_vazia(pilha)) ):
        while arvore_aux != None:
            pilha.append(arvore_aux)
            weight = arvore_aux.weight
            arvore_aux = arvore_aux.left
            set_weight(arvore_aux, weight)
        arvore_aux = pilha.pop()
        if arvore_aux.right == None and arvore_aux.left == None:#Se no for folha
          vet_weight.append(arvore_aux.weight); tree_leaves.append(arvore_aux.data)
        weight = arvore_aux.weight
        arvore_aux = arvore_aux.right
        set_weight(arvore_aux, weight)
    return vet_weight, tree_leaves

def pilha_vazia(pilha):
    '''
    Função auxiliar que serve somente para sabermos se a stack esta vazia.

    Parameters
    ----------
    pilha : list
        Pilha é uma lista que esta sendo usada como stack

    Return
    len(pilha) : bool
        Retorna verdadeiro caso a pilha esteja vazia, ou falso caso tenha pelo menos 1 elemento
    '''
    return len(pilha) == 0

def set_weight(arvore, weight):
    '''
    Função auxiliar para atribuir o peso para o novo cluster ou para um ativo caso seja um nó folha.
    Nesse caso como estamos seguindo o paper de Thomas Raffinot apenas pegamos o peso do nó acima na
    arvore e dividimos por 2 para termos o novo peso.

    Parameters
    arvore : Tree
        raiz da arvore que usaremos para guardar o novo peso
    weight : float
        Peso do nó acima na arvore do nó que recebemos. Dessa forma, conseguimos calcular o peso para o nó arvore recebido,
        como no caso do paper do Thomas Raffinot ele usa o sistema de peso igual entre os clusters, aqui apenas pegamos o
        peso do nó acima e dividimos por 2.
    '''
    if arvore != None: arvore.weight = weight / 2

def fstage_hc(data_stocks):
    '''
    Essa é uma função casca para aglomerar uma série de comandos, nesse caso os comandos estão
    relacionados ao primeiro estagio do processo de criar o portfolio que é o processo de
    Hierarchical clustering

    Parameters
    ----------
    data_stocks : dataframe pandas
        É a matriz obtida através da API do Yahoo e contem todos os dados dos ativos passados

    Return
    ------
    clustering_matrix: ndarray
        A matriz de hierarchical clustering codificada como uma matriz de link
    '''
    # Stage 1: Hierarchical Clustering
    correlation = get_correlation(data_stocks)
    distance = calc_distance(correlation)
    euclidean_distance = squareform(pdist(distance, metric='euclidean'))
    clustering = hierarchical_clustering(euclidean_distance, 'ward')
    return clustering

def adc_stage(clustering, cutoff):
    '''
    Essa é uma função casca para aglomerar uma série de comandos, nesse caso os comandos estão
    relacionados ao estagio adicional do processo de criação do portfolio que é a criação da
    arvore para nos ajudar a dar peso para os ativos de forma mais facil.

    Parameters
    ----------
    clustering_matrix : ndarray
        A matriz de hierarchical clustering codificada como uma matriz de link
    cutoff : float
        valor que representa o ponto de cutoff, ou seja, distancias entre cada cluster combinado
        que for menor do que esse valor deve ser removido

    Returns
    ------
    raiz : Tree
        Essa arvore criada nos auxiliará no processo de dar peso aos ativos e clusters. Essa arvore
        tem também como objetivo representar o dendrogram
    new_dicio : dict
        Caso tenha sido fornecido um ponto de cutoff retornaremos então new_dicio e não percorreremos
        o restante da função para não executar os outros comandos sem necessidade, pois dentro da
        função new_dict_cutoff esses passos já foram feitos
    '''
    #dicionario contendo os indices dos clusters gerados da combinação de outros 2
    #cluster_merged = get_idx_cluster_merged(clustering, len_asset) #sem cutoff
    cluster_merged = get_idx_cluster_merged(clustering, len(clustering)+1, None) #sem cutoff
    if cutoff != None:
        asset_weight, new_dicio = new_dict_cutoff(clustering, cluster_merged, cutoff)
        return new_dicio
    #lista contendo as chaves do dicionario
    keys = list(cluster_merged.keys())
    #ultimos clusters merged
    cluster = cluster_merged[keys[-1]]
    #Cria arvore onde a raiz é os dois ultimos clusters combinados
    raiz = Tree(cluster, 100)
    #criando a arvore atravez dos dois ultimos clusters combinados
    #usando o dicionario para mapear os clusters que foram criados atraves da combinação
    #passando a arvore para prencher o campo data de cada no
    create_tree(cluster, cluster_merged, raiz, None)
    return raiz

def sstage_weight(raiz):
    '''
    Essa é uma função casca para aglomerar uma série de comandos, nesse caso os comandos estão
    relacionados ao segundo estagio que é o processo de dar peso aos ativos e clusters.

    Parameters
    ----------
    raiz : Tree
        raiz da arvore que criada que será usada agora para dar peso aos nós

    Return
    ------
    asset_weight : dict
        Dicionario contendo o numero do ativo como chave e seu peso no portfolio como valor
    '''
    #vetor com o peso de cada ativo e cluster
    vet_weight, leaves_tree = weight_tree(raiz)
    #dicionario mapeando o ativo e seu respectivo peso no portfolio
    asset_weight = {key: f'{value}%' for key, value in zip(leaves_tree, vet_weight)}
    return asset_weight

def new_dict_cutoff(clustering, dicio, cutoff):
    '''
    Essa função tem como objetivo final calcular o novo dicionario com os ativos e seus respectivos pesos


    Parameters
    clustering_matrix : ndarray
        A matriz de hierarchical clustering codificada como uma matriz de link
    dicio : dict
        Dicionario onde a chave é o indice dos clusters que foram criados através da combinação de outros
        clusters, e os valores são os indices desses clusters
    cutoff : float
        Valor que usaremos para cortar clusters cuja distancia é menor do que o valor de cutoff fornecido

    Return
    ------
    rebuild_dict : tuple
        Tupla contendo o dicionario asset_weight que é o dicionario criado com base no ponto de cutoff
        e new_dicio que é o dicionario calculado com base no asset_weight.
    '''
    #pegando o ultimo cluster a ser menor que o ponto de cutoff
    cutoff_pos = clustering[0]; last_cluster = 10
    for i in range(len(clustering)):
        if clustering[:,2][i] >= cutoff: break
        cutoff_pos = clustering[i]
        last_cluster = 10 + i
    #Processo de pegar as folhas do cluster a ser removido
    #values = ultimo cluster a ter a distancia <= cutoff
    values = [int(cutoff_pos[0]), int(cutoff_pos[1])]
    #folhas que seram atribuidos ao ultimo cluster <= cutoff
    leaves = get_leaves(values, dicio)
    dicio_cutoff = get_idx_cluster_merged(clustering, last_cluster, leaves)
    keys = list(dicio_cutoff.keys())
    cluster = dicio_cutoff[keys[-1]]
    raiz = Tree(cluster, 100)
    cluster = create_tree(cluster, dicio_cutoff, raiz, last_cluster)
    vet_weight, leaves_tree = weight_tree(raiz)
    asset_weight = {key: f'{value}%' for key, value in zip(leaves_tree, vet_weight)}
    return rebuild_dict(asset_weight, dicio)

def rebuild_dict(asset_weight, dicio):
    '''
    Essa função serve para criar o novo dicionario contendo os ativos como chave e os valores sendo o peso de
    cada ativo.
    Essa função só sera usada caso tenha algum ponto de cutoff, assim precisamos recriar o dicionario de ativos
    e peso.

    Parameters
    ---------
    asset_weight : dict
        Dicionario que foi feito com base no ponto de cutoff
    dicio : dict
        Dicionario que foi feito sem que um ponto de cutoff fosse passado

    Return
    ------
    asset_weight : dict
        Dicionario que foi feito com base no ponto de cutoff
    new_dicio : dict
        Esse novo dicionario foi feito com base no ponto de cutoff e no dicionario asset_weight
        usando o dicionario asset_weight substituimos os valores maiores que 10 que são os clusters
        contendo varias folhas pelas proprias folhas e atribuimos os pesos a elas, esse peso é calculado
        com base no peso que antes era do cluster inteiro, ou seja, dividimos pela quantidade de ativos
        que estavam de certa forma 'escondidos' ali.
    '''
    keys = list(asset_weight.keys())
    print(asset_weight)
    new_dicio = {}
    for key in keys:
        weight = float(asset_weight[key].replace("%", ''))
        if str(key) in dicio and key >= 10:
            leaves = get_leaves(dicio[f'{key}'], dicio)
            weight_per_leaf = np.round(weight / len(leaves), decimals=4)
            new_dicio.update({ leaf: f'{weight_per_leaf}%' for leaf in leaves})
        else:
            new_dicio[key] = f'{weight}%'
    return asset_weight, new_dicio

def get_leaves(values, dicio):
    '''
    Essa função nos auxiliará a achar as folhas de values para quando um ponto de cutoff for passado.
    Assim usamos essa função para encontrar as folhas que agora vão pertecer a 1 só cluster.

    Parameters
    ----------
    values : list
        Valores do cluster para buscarmos suas folhas, ou seja, os valores de outros clusters que fazem parte.
    dicio : dict
        Dicionario contendo como chave os indices dos clusters criados a partir de outros dois, e os valores são esses clusters

    Return
    ------
    result : list
        Lista contendo todas as folhas que foram encontradas a partir de values
    '''
    result = []
    for val in values:
      if val >= 10:
        result.extend(get_leaves(dicio[f"{val}"], dicio))
      else:
        result.append(val)
    return result

def portfolio(cutoff):
    '''
    Essa é uma função casca para aglomerar uma série de comandos, nesse caso os comandos estão
    relacionados ao processo de criação de fato do portfolio, aglomerando assim os comandos para
    os estagios, primeiro, adicional e segundo.

    Parameters
    ----------
    cutoff : float
        valor que representa o ponto de cutoff, ou seja, distancias entre cada cluster combinado
        que for menor do que esse valor deve ser removido
    '''
    asset = ['MSFT', 'PCAR', 'JPM', 'AAPL', 'GOOGL', 'AMZN', 'ITUB', 'VALE', 'SHEL', 'INTC']
    start = '2016-01-01'; end = '2022-01-01'
    data_stocks = get_stocks(asset, start,  end)
    # Stage 1: Hierarchical Clustering
    clustering = fstage_hc(data_stocks)
    # Additional Stage to aux the weight stage
    tree_or_dict = adc_stage(clustering, cutoff)
    #Stage 2: Assigning weights to clusters
    dict_asset_weight = sstage_weight(tree_or_dict) if cutoff == None else tree_or_dict
    print(dict_asset_weight)

def main():
    portfolio(None)
    #portfolio(0.5)
    #portfolio(1.0)

if __name__ == "__main__":
    main()