// I B E X
// File : ibex_Optimizer.cpp
// Author : Gilles Chabert, Bertrand Neveu
// Copyright : IMT Atlantique (France)
// License : See the LICENSE file
// Created : May 14, 2012
// Last Update : Feb 13, 2025
//============================================================================
#include "ibex_Optimizer.h"
#include "ibex_Timer.h"
#include "ibex_Function.h"
#include "ibex_NoBisectableVariableException.h"
#include "ibex_BxpOptimData.h"
#include "ibex_CovOptimData.h"
#include <float.h>
#include <stdlib.h>
#include <iomanip>
#include <fstream>	 	 // Para ofstream/*
#include <string>		 // Para std::to_string
#include <algorithm> 	 // Para std::min (si lo necesitas en otra parte)
#include <map>       	 // Para std::map (HDBSCAN)
#include <unordered_map> // Para std::unordered_map (HDBSCAN)
#include <queue>	  	 // Para cola de prioridad (HDBSCAN)
#include <cmath>
#include <iostream> // Para std::cout, std::endl si los usas directamente

using namespace std;
// ——————————————————————————————————————————————————————————————————————
// Código de clustering embebido
#include <vector>
#include <random>
#include <limits>

// NORMALIZACION
static std::vector<double> inv_range;        // 1/diámetro por dimensión
static ibex::IntervalVector root_box_norm;   // copia de la caja raíz
// FIN NORMALIZACION


// Representa el centro de cada caja (dim = n+1)
using Point = std::vector<double>;
// Resultado de k-means: etiqueta por punto y nº de clústeres
struct ClusterResult
{
	std::vector<int> labels;
	int n_clusters;
};

/// Versión simple de K-Means++
static ClusterResult kmeans_plus(const std::vector<Point> &data, int k)
{
    int n = data.size();
    int dim = data[0].size();
    std::mt19937 gen(0);
    
    //INICIALIZACIÓN K-Means++
    std::vector<Point> centroids;
    centroids.reserve(k);
    
    // Paso 1: Seleccionar primer centroide aleatoriamente
    std::uniform_int_distribution<> dis_first(0, n - 1);
    centroids.push_back(data[dis_first(gen)]);
    
    // Vector de distancias mínimas al centroide más cercano
    std::vector<double> min_distances(n, std::numeric_limits<double>::infinity());
    
    // Paso 2: Seleccionar k-1 centroides restantes con K-Means++
    for (int c = 1; c < k; ++c) {
        double total_distance = 0.0;
        
        // Calcular D(x)² para cada punto
        for (int i = 0; i < n; ++i) {
            // Distancia al último centroide agregado
            double d = 0.0;
            for (int d0 = 0; d0 < dim; ++d0) {
                double diff = data[i][d0] - centroids.back()[d0];
                d += diff * diff;
            }
            
            // Actualizar distancia mínima si es necesario
            if (d < min_distances[i]) {
                min_distances[i] = d;
            }
            
            total_distance += min_distances[i];
        }
        
        // Selección probabilística: P(x) ∝ D(x)²
        if (total_distance == 0) {
            std::uniform_int_distribution<> dis_fallback(0, n - 1);
            centroids.push_back(data[dis_fallback(gen)]);
        } else {
            std::uniform_real_distribution<> dis_prob(0.0, total_distance);
            double target = dis_prob(gen);
            
            double cumulative = 0.0;
            for (int i = 0; i < n; ++i) {
                cumulative += min_distances[i];
                if (cumulative >= target) {
                    centroids.push_back(data[i]);
                    break;
                }
            }
        }
    }
    
    // ALGORITMO K-MEANS ESTÁNDAR CON EARLY STOPPING
    std::vector<int> labels(n, 0);
    bool changed = true;
    
    // ── EARLY STOPPING: Variable para rastrear inercia anterior ──
    double prev_inertia = std::numeric_limits<double>::infinity();
    const double convergence_threshold = 0.001; // 0.1% de cambio
    
    for (int iter = 0; iter < 50 && changed; ++iter)
    {
        changed = false;
        
        // Paso de Asignación
        for (int i = 0; i < n; ++i)
        {
            double best = std::numeric_limits<double>::infinity();
            int bi = 0;
            
            for (int c = 0; c < k; ++c)
            {
                double d = 0;
                for (int d0 = 0; d0 < dim; ++d0)
                {
                    double diff = data[i][d0] - centroids[c][d0];
                    d += diff * diff;
                }
                if (d < best)
                {
                    best = d;
                    bi = c;
                }
            }
            
            if (labels[i] != bi)
            {
                labels[i] = bi;
                changed = true;
            }
        }
        
        // Paso de Re-cálculo de centroides
        std::vector<Point> sum(k, Point(dim, 0.0));
        std::vector<int> cnt(k, 0);
        
        for (int i = 0; i < n; ++i)
        {
            cnt[labels[i]]++;
            for (int d0 = 0; d0 < dim; ++d0)
                sum[labels[i]][d0] += data[i][d0];
        }
        
        for (int c = 0; c < k; ++c)
        {
            if (cnt[c] > 0)
                for (int d0 = 0; d0 < dim; ++d0)
                    centroids[c][d0] = sum[c][d0] / cnt[c];
            else
                centroids[c] = data[dis_first(gen)];
        }
        
        // EARLY STOPPING: Calcular inercia (suma de distancias²)
        double inertia = 0.0;
        for (int i = 0; i < n; ++i) {
            double d_sq = 0.0;
            for (int d0 = 0; d0 < dim; ++d0) {
                double diff = data[i][d0] - centroids[labels[i]][d0];
                d_sq += diff * diff;
            }
            inertia += d_sq;
        }
        
        // Verificar convergencia: si el cambio es < 0.1% de la inercia anterior
        if (std::abs(inertia - prev_inertia) < convergence_threshold * prev_inertia) {
            // std::cout << "[K-Means++] Convergencia en iteración " << iter + 1 << std::endl;
            break; // SALIR ANTICIPADAMENTE
        }
        
        prev_inertia = inertia;
    }
    
    return ClusterResult{labels, k};
}

// ══════════════════════════════════════════════════════════════════════════════
// HDBSCAN: Hierarchical Density-Based Spatial Clustering
// ══════════════════════════════════════════════════════════════════════════════

// Resultado del clustering: etiquetas (-1 = ruido) y número de clústeres
struct HdbscanClusterResult {
    std::vector<int> labels;
    int num_clusters;
};

// Arista del MST: conecta nodos u-v con peso (mutual reachability distance)
struct Edge {
    int u, v;
    double weight;
    bool operator>(const Edge& other) const {
        return weight > other.weight;
    }
};

// Union-Find con compresión de caminos y unión por rango
// Mantiene componentes conexas durante la construcción de jerarquía
class UnionFind {
private:
    std::vector<int> parent;  // parent[i] = representante del conjunto de i
    std::vector<int> rank;    // rank[i] = altura aproximada del subárbol
public:
    UnionFind(int n) : parent(n), rank(n, 0) {
        for (int i = 0; i < n; ++i)
            parent[i] = i;  // Inicialmente cada nodo es su propio representante
    }
    
    // Encuentra la raíz del conjunto que contiene x (con compresión de caminos)
    int find(int x) {
        if (parent[x] != x)
            parent[x] = find(parent[x]);
        return parent[x];
    }
    
    // Une los conjuntos de x e y (retorna true si se realizó la unión)
    bool unite(int x, int y) {
        int px = find(x);
        int py = find(y);
        if (px == py) return false;
        
        // Unión por rango: el árbol de menor altura se cuelga del de mayor altura
        if (rank[px] < rank[py]) {
            parent[px] = py;
        } else if (rank[px] > rank[py]) {
            parent[py] = px;
        } else {
            parent[py] = px;
            rank[px]++;
        }
        return true;
    }
};

// Distancia euclidiana entre puntos multidimensionales
static double hdbscan_distance(const Point& p1, const Point& p2) {
    double sum = 0.0;
    for (size_t i = 0; i < p1.size(); ++i) {
        double diff = p1[i] - p2[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// Calcula core distance de cada punto: distancia al k-ésimo vecino más cercano
static std::vector<double> compute_core_distances(
    const std::vector<Point>& data, 
    int min_samples) 
{
    int n = data.size();
    std::vector<double> core_dist(n);
    
    // Validar min_samples (previene accesos fuera de rango)
    if (min_samples <= 0 || min_samples > n) {
        if (n > 0)
            min_samples = std::min(n, 15);
        else
            return core_dist;
    }
    
    std::vector<double> dists;
    dists.reserve(n - 1);
    
    for (int i = 0; i < n; ++i) {
        dists.clear();
        
        // Calcular distancias a todos los demás puntos
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            dists.push_back(hdbscan_distance(data[i], data[j]));
        }
        
        // Encontrar k-ésima distancia más pequeña con nth_element (O(n) promedio)
        if (dists.size() >= (size_t)min_samples) {
            std::nth_element(dists.begin(),
                           dists.begin() + (min_samples - 1),
                           dists.end());
            core_dist[i] = dists[min_samples - 1];
        } else if (!dists.empty()) {
            core_dist[i] = *std::max_element(dists.begin(), dists.end());
        }
    }
    
    return core_dist;
}

// Construcción de MST con algoritmo de Prim usando mutual reachability distances
static std::vector<Edge> build_mst(
    const std::vector<Point>& data,
    const std::vector<double>& core_distances,
    int max_neighbors = 30)
{
    int n = data.size();
    std::vector<Edge> mst_edges;
    mst_edges.reserve(n - 1);  // MST de n nodos tiene n-1 aristas
    
    std::vector<bool> in_mst(n, false);
    using PQElement = std::tuple<double, int, int>;  // (distancia, destino, origen)
    std::priority_queue<PQElement, std::vector<PQElement>, std::greater<PQElement>> pq;
    
    in_mst[0] = true;  // Iniciar desde nodo 0
    
    // Fase inicial: agregar aristas desde nodo 0 a sus K vecinos más cercanos
    std::vector<std::pair<double, int>> initial_neighbors;
    for (int v = 1; v < n; ++v) {
        double dist_0v = hdbscan_distance(data[0], data[v]);
        double mutual_dist = std::max(dist_0v, std::max(core_distances[0], core_distances[v]));
        initial_neighbors.push_back({mutual_dist, v});
    }
    
    int K = std::min((int)initial_neighbors.size(), max_neighbors);
    if (K > 0) {
        std::partial_sort(initial_neighbors.begin(), 
                        initial_neighbors.begin() + K, 
                        initial_neighbors.end());
        for (int i = 0; i < K; ++i) {
            pq.push({initial_neighbors[i].first, initial_neighbors[i].second, 0});
        }
    }
    
    // Algoritmo de Prim: expandir MST seleccionando arista de menor peso
    while (!pq.empty() && (int)mst_edges.size() < n - 1) {
        auto [dist, u, parent] = pq.top();
        pq.pop();
        
        if (in_mst[u]) continue;
        in_mst[u] = true;
        
        mst_edges.push_back({parent, u, dist});
        
        // Expandir desde u: agregar aristas a nodos no visitados
        std::vector<std::pair<double, int>> candidates;
        for (int v = 0; v < n; ++v) {
            if (!in_mst[v]) {
                double dist_uv = hdbscan_distance(data[u], data[v]);
                double mutual_dist = std::max(dist_uv, std::max(core_distances[u], core_distances[v]));
                candidates.push_back({mutual_dist, v});
            }
        }
        
        // Limitar a K mejores candidatos (reduce tamaño del heap)
        K = std::min((int)candidates.size(), max_neighbors);
        if (K > 0) {
            std::partial_sort(candidates.begin(), 
                            candidates.begin() + K, 
                            candidates.end());
            
            for (int i = 0; i < K; ++i) {
                pq.push({candidates[i].first, candidates[i].second, u});
            }
        }
    }
    
    return mst_edges;
}

// Extracción de clústeres del MST mediante corte incremental
static HdbscanClusterResult extract_clusters(
    const std::vector<Edge>& mst_edges,
    int n,
    int min_cluster_size)
{
    HdbscanClusterResult result;
    result.labels.assign(n, -1);  // Inicialmente todos son ruido
    
    if (mst_edges.empty()) {
        result.num_clusters = 0;
        return result;
    }
    
    // Ordenar aristas por peso descendente (cortar de mayor a menor peso)
    std::vector<Edge> sorted_edges = mst_edges;
    std::sort(sorted_edges.begin(), sorted_edges.end(),
              [](const Edge& a, const Edge& b) { return a.weight > b.weight; });
    
    UnionFind uf(n);
    std::vector<int> cluster_size(n, 1);
    std::map<int, int> root_to_label;
    int current_label = 0;
    
    // Proceso de corte: unir componentes según orden de similaridad
    for (const auto& edge : sorted_edges) {
        int root_u = uf.find(edge.u);
        int root_v = uf.find(edge.v);
        
        // Verificar si componentes han alcanzado tamaño mínimo
        bool u_is_cluster = (cluster_size[root_u] >= min_cluster_size);
        bool v_is_cluster = (cluster_size[root_v] >= min_cluster_size);
        
        // Asignar etiquetas a componentes que son clústeres válidos
        if (u_is_cluster && root_to_label.find(root_u) == root_to_label.end()) {
            root_to_label[root_u] = current_label++;
        }
        if (v_is_cluster && root_to_label.find(root_v) == root_to_label.end()) {
            root_to_label[root_v] = current_label++;
        }
        
        // Unir componentes y actualizar tamaño
        if (root_u != root_v) {
            uf.unite(root_u, root_v);
            int new_root = uf.find(root_u);
            cluster_size[new_root] = cluster_size[root_u] + cluster_size[root_v];
        }
    }
    
    // Asignar etiquetas finales (ruido permanece como -1)
    for (int i = 0; i < n; ++i) {
        int root = uf.find(i);
        if (root_to_label.find(root) != root_to_label.end()) {
            result.labels[i] = root_to_label[root];
        }
    }
    
    result.num_clusters = current_label;
    return result;
}

// Función principal de HDBSCAN
static HdbscanClusterResult hdbscan(
    const std::vector<Point>& data,
    int min_cluster_size,
    int min_samples = -1)
{
    int n = data.size();
    
    if (min_samples == -1) {
        min_samples = min_cluster_size;
    }
    
    HdbscanClusterResult result;
    
    // Validaciones
    if (n == 0 || min_cluster_size <= 0) {
        result.labels.assign(n, -1);
        result.num_clusters = 0;
        return result;
    }
    
    if (n < min_cluster_size) {
        result.labels.assign(n, -1);
        result.num_clusters = 0;
        return result;
    }
    
    // Paso 1: Calcular core distances (k-ésimo vecino)
    std::vector<double> core_distances = compute_core_distances(data, min_samples);
    
    // Paso 2: Construir MST con mutual reachability distances
    int max_neighbors = std::max(min_samples * 2, 30);
    std::vector<Edge> mst = build_mst(data, core_distances, max_neighbors);
    
    // Paso 3: Extraer clústeres por corte jerárquico
    result = extract_clusters(mst, n, min_cluster_size);
    
    return result;
}

// ══════════════════════════════════════════════════════════════════════════════════════


namespace {

double calculate_box_volume(const ibex::IntervalVector& box) {
    if (box.is_empty()) {
        return 0.0;
    }
    if (box.size() == 0) { // Caja 0-dimensional
        return 1.0; // Convencionalmente, el volumen de un punto (o el producto vacío) es 1.
                    // Si prefieres 0.0 para una caja sin dimensiones, cámbialo.
    }

    double volume = 1.0;
    bool has_infinite_dimension = false;

    for (int i = 0; i < box.size(); ++i) {
        const ibex::Interval& component = box[i];
        
        if (component.is_empty()) { // Si cualquier componente es vacío, el volumen total es 0
            return 0.0;
        }

        double diam = component.diam();

        if (diam < 0) { // Diámetro negativo implica vacío para Ibex
            return 0.0;
        }

		if (diam == std::numeric_limits<double>::infinity()) {
			has_infinite_dimension = true;
		} else if (diam == 0.0) {
            // Si cualquier dimensión finita tiene diámetro 0, el volumen total es 0.
            // Esto tiene prioridad sobre una dimensión infinita (ej. una línea en un plano tiene volumen 0).
            return 0.0;
        } else {
            // Solo multiplica diámetros finitos y no cero por ahora.
            volume *= diam; 
            // Comprobación temprana de overflow a infinito si el volumen ya es enorme
            if (volume == std::numeric_limits<double>::infinity()) break; 
        }
    }

    // Si hubo una dimensión infinita Y ninguna dimensión con diámetro cero, el volumen es infinito.
    if (has_infinite_dimension) {
        return std::numeric_limits<double>::infinity();
    }

    return volume;
}

} // fin del namespace anónimo

namespace ibex
{
	/*
	 * TODO: redundant with ExtendedSystem.
	 */
	void Optimizer::write_ext_box(const IntervalVector &box, IntervalVector &ext_box)
	{
		int i2 = 0;
		for (int i = 0; i < n; i++, i2++)
		{
			if (i2 == goal_var)
				i2++; // skip goal variable
			ext_box[i2] = box[i];
		}
	}
	void Optimizer::read_ext_box(const IntervalVector &ext_box, IntervalVector &box)
	{
		int i2 = 0;
		for (int i = 0; i < n; i++, i2++)
		{
			if (i2 == goal_var)
				i2++; // skip goal variable
			box[i] = ext_box[i2];
		}
	}
	Optimizer::Optimizer(int n, Ctc &ctc, Bsc &bsc, LoupFinder &finder,
						 CellBufferOptim &buffer,
						 int goal_var, double eps_x, double rel_eps_f, double abs_eps_f,
						 bool enable_statistics) : n(n), goal_var(goal_var),
												   ctc(ctc), bsc(bsc), loup_finder(finder), buffer(buffer),
												   eps_x(n, eps_x), rel_eps_f(rel_eps_f), abs_eps_f(abs_eps_f),
												   trace(0), timeout(-1), extended_COV(true), anticipated_upper_bounding(true),
												   status(SUCCESS),
												   uplo(NEG_INFINITY), uplo_of_epsboxes(POS_INFINITY), loup(POS_INFINITY),
												   loup_point(IntervalVector::empty(n)), initial_loup(POS_INFINITY), loup_changed(false),
												   time(0), nb_cells(0), cov(NULL), clustering_params()
	{
		if (trace)
			cout.precision(12);

		// Inicialización del control de reinicios

		clustering_params.choice = ClusteringParams::Algorithm::KMEANS_PLUS; // Por defecto KMEANS_PLUS

		restart_threshold = 500; // Umbral de iteraciones sin mejora de la cota superior
		node_threshold = 1000;   // Umbral de tamaño máximo de nodos en el buffer
		stagnation_counter = 0;

		buffer_size_before_restart = 0; 		// Tamaño del buffer antes del reinicio
    	loup_before_restart = POS_INFINITY; 	// Valor de loup antes del reinicio

		clustering_params.hull_volume_threshold_fraction = 3.0; // 3 para tener maximo de 3 veces mas la caja inicial

		// HDBSCAN
        clustering_params.hdbscan_min_cluster_size = 15;  // Tamaño mínimo del cluster por defecto
        clustering_params.hdbscan_min_samples = -1;  	  // Usar min_cluster_size por defecto

		if (enable_statistics)
		{
			statistics = new Statistics();
			// TODO: enable statistics for missing operators (cell buffer)
			bsc.enable_statistics(*statistics, "Bsc");
			ctc.enable_statistics(*statistics, "Ctc");
			loup_finder.enable_statistics(*statistics, "LoupFinder");
		}
		else
			statistics = NULL;
	}
	Optimizer::Optimizer(OptimizerConfig &config) : Optimizer(
														config.nb_var(),
														config.get_ctc(),
														config.get_bsc(),
														config.get_loup_finder(),
														config.get_cell_buffer(),
														config.goal_var(),
														OptimizerConfig::default_eps_x, // tmp, see below
														config.get_rel_eps_f(),
														config.get_abs_eps_f(),
														config.with_statistics())
	{
		(Vector &)eps_x = config.get_eps_x();
		trace = config.get_trace();
		timeout = config.get_timeout();
		extended_COV = config.with_extended_cov();
		anticipated_upper_bounding = config.with_anticipated_upper_bounding();
	}
	Optimizer::~Optimizer()
	{
		if (cov)
			delete cov;
		if (statistics)
			delete statistics;
	}
	// compute the value ymax (decreasing the loup with the precision)
	// the heap and the current box are contracted with y <= ymax
	double Optimizer::compute_ymax()
	{
		if (anticipated_upper_bounding)
		{
			// double ymax = loup - rel_eps_f*fabs(loup); ---> wrong :the relative precision must be correct for ymax (not loup)
			double ymax = loup > 0 ? 1 / (1 + rel_eps_f) * loup
								   : 1 / (1 - rel_eps_f) * loup;
			if (loup - abs_eps_f < ymax)
				ymax = loup - abs_eps_f;
			// return ymax;
			return next_float(ymax);
		}
		else
			return loup;
	}
	bool Optimizer::update_loup(const IntervalVector &box, BoxProperties &prop)
	{
		try
		{
			pair<IntervalVector, double> p = loup_finder.find(box, loup_point, loup, prop);
			loup_point = p.first;
			loup = p.second;
			if (trace)
			{
				cout << " ";
				cout << "\033[32m loup= " << loup << "\033[0m" << endl;
				// cout << " loup point=";
				// if (loup_finder.rigorous())
				// cout << loup_point << endl;
				// else
				// cout << loup_point.lb() << endl;
			}
			return true;
		}
		catch (LoupFinder::NotFound &)
		{
			return false;
		}
	}
	// bool Optimizer::update_entailed_ctr(const IntervalVector& box) {
	// for (int j=0; j<m; j++) {
	// if (entailed->normalized(j)) {
	// continue;
	// }
	// Interval y=sys.ctrs[j].f.eval(box);
	// if (y.lb()>0) return false;
	// else if (y.ub()<=0) {
	// entailed->set_normalized_entailed(j);
	// }
	// }
	// return true;
	//}
	void Optimizer::update_uplo()
	{
		double new_uplo = POS_INFINITY;
		if (!buffer.empty())
		{
			new_uplo = buffer.minimum();
			if (new_uplo > loup && uplo_of_epsboxes > loup)
			{
				cout << " loup = " << loup << " new_uplo=" << new_uplo << " uplo_of_epsboxes=" << uplo_of_epsboxes << endl;
				ibex_error("optimizer: new_uplo>loup (please report bug)");
			}
			if (new_uplo < uplo)
			{
				cout << "uplo= " << uplo << " new_uplo=" << new_uplo << endl;
				ibex_error("optimizer: new_uplo<uplo (please report bug)");
			}
			// uplo <- max(uplo, min(new_uplo, uplo_of_epsboxes))
			if (new_uplo < uplo_of_epsboxes)
			{
				if (new_uplo > uplo)
				{
					uplo = new_uplo;
					if (trace)
						cout << "\033[33m uplo= " << uplo << "\033[0m" << endl;
				}
			}
			else
				uplo = uplo_of_epsboxes;
		}
		else if (buffer.empty() && loup != POS_INFINITY)
		{
			// empty buffer : new uplo is set to ymax (loup - precision) if a loup has been found
			new_uplo = compute_ymax(); // not new_uplo=loup, because constraint y <= ymax was enforced
			// cout << " new uplo buffer empty " << new_uplo << " uplo " << uplo << endl;
			double m = (new_uplo < uplo_of_epsboxes) ? new_uplo : uplo_of_epsboxes;
			if (uplo < m)
				uplo = m; // warning: hides the field "m" of the class
						  // note: we always have uplo <= uplo_of_epsboxes but we may have uplo > new_uplo, because
						  // ymax is strictly lower than the loup.
		}
	}
	void Optimizer::update_uplo_of_epsboxes(double ymin)
	{
		// the current box cannot be bisected. ymin is a lower bound of the objective on this box
		// uplo of epsboxes can only go down, but not under uplo : it is an upperbound for uplo,
		// that indicates a lowerbound for the objective in all the small boxes
		// found by the precision criterion
		assert(uplo_of_epsboxes >= uplo);
		assert(ymin >= uplo);
		if (uplo_of_epsboxes > ymin)
		{
			uplo_of_epsboxes = ymin;
			if (trace)
			{
				cout << " unprocessable tiny box: now uplo<=" << setprecision(12) << uplo_of_epsboxes << " uplo=" << uplo << endl;
			}
		}
	}
	void Optimizer::handle_cell(Cell &c)
	{
		contract_and_bound(c);
		if (c.box.is_empty())
		{
			delete &c;
		}
		else
		{
			buffer.push(&c);
		}
	}
	void Optimizer::contract_and_bound(Cell &c)
	{
		/*======================== contract y with y<=loup ========================*/
		Interval &y = c.box[goal_var];
		double ymax;
		if (loup == POS_INFINITY)
			ymax = POS_INFINITY;
		// ymax is slightly increased to favour subboxes of the loup
		// TODO: useful with double heap??
		else
			ymax = compute_ymax() + 1.e-15;
		y &= Interval(NEG_INFINITY, ymax);
		if (y.is_empty())
		{
			c.box.set_empty();
			return;
		}
		else
		{
			c.prop.update(BoxEvent(c.box, BoxEvent::CONTRACT, BitSet::singleton(n + 1, goal_var)));
		}
		/*================ contract x with f(x)=y and g(x)<=0 ================*/
		// cout << " [contract] x before=" << c.box << endl;
		// cout << " [contract] y before=" << y << endl;
		ContractContext context(c.prop);
		if (c.bisected_var != -1)
		{
			context.impact.clear();
			context.impact.add(c.bisected_var);
			context.impact.add(goal_var);
		}
		ctc.contract(c.box, context);
		// cout << c.prop << endl;
		if (c.box.is_empty())
			return;
		// cout << " [contract] x after=" << c.box << endl;
		// cout << " [contract] y after=" << y << endl;
		/*====================================================================*/
		/*========================= update loup =============================*/
		IntervalVector tmp_box(n);
		read_ext_box(c.box, tmp_box);
		c.prop.update(BoxEvent(c.box, BoxEvent::CHANGE));
		bool loup_ch = update_loup(tmp_box, c.prop);
		// update of the upper bound of y in case of a new loup found
		if (loup_ch)
		{
			y &= Interval(NEG_INFINITY, compute_ymax());
			c.prop.update(BoxEvent(c.box, BoxEvent::CONTRACT, BitSet::singleton(n + 1, goal_var)));
		}
		// TODO: should we propagate constraints again?
		loup_changed |= loup_ch;
		if (y.is_empty())
		{ // fix issue #44
			c.box.set_empty();
			return;
		}
		/*====================================================================*/
		// Note: there are three different cases of "epsilon" box,
		// - NoBisectableVariableException raised by the bisector (---> see optimize(...)) which
		// is independent from the optimizer
		// - the width of the box is less than the precision given to the optimizer ("eps_x" for
		// the original variables and "abs_eps_f" for the goal variable)
		// - the extended box has no bisectable domains (if eps_x=0 or <1 ulp)
		if (((tmp_box.diam() - eps_x).max() <= 0 && y.diam() <= abs_eps_f) || !c.box.is_bisectable())
		{
			update_uplo_of_epsboxes(y.lb());
			c.box.set_empty();
			return;
		}
		// ** important: ** must be done after upper-bounding
		// kkt.contract(tmp_box);
		if (tmp_box.is_empty())
		{
			c.box.set_empty();
		}
		else
		{
			// the current extended box in the cell is updated
			write_ext_box(tmp_box, c.box);
		}
	}
	Optimizer::Status Optimizer::optimize(const IntervalVector &init_box, double obj_init_bound)
	{

		/*
		// **NUEVO**: Mostrar información y volumen de la caja inicial (espacio de decisión)
		if (trace > 0 && !init_box.is_empty() && init_box.size() == this->n) {
		// Asumimos que 'this->n' es el número de variables originales del problema
		cout << "[Optimizer] Problema con " << this->n << " variables de decisión." << endl;
		cout << "[Optimizer] Caja inicial (espacio de decisión): " << init_box << endl;
		double initial_volume = init_box.volume();
		if (initial_volume == POS_INFINITY && init_box.max_diam() == POS_INFINITY) {
		cout << "[Optimizer] Volumen de caja inicial (espacio de decisión): Infinito (una o más dimensiones son no acotadas)" << endl;
		} else {
		cout << "[Optimizer] Volumen de caja inicial (espacio de decisión): " << initial_volume << endl;
		}
		}
		// --- FIN DE LA MODIFICACIÓN ---
		*/
		start(init_box, obj_init_bound);
		return optimize();
	}
	Optimizer::Status Optimizer::optimize(const CovOptimData &data, double obj_init_bound)
	{
		start(data, obj_init_bound);
		return optimize();
	}
	Optimizer::Status Optimizer::optimize(const char *cov_file, double obj_init_bound)
	{
		CovOptimData data(cov_file);
		start(data, obj_init_bound);
		return optimize();
	}
	void Optimizer::start(const IntervalVector &init_box, double obj_init_bound)
	{
		loup = obj_init_bound;
		// Just to initialize the "loup" for the buffer
		// TODO: replace with a set_loup function

		/* -------- Normalización: cachear la caja inicial -------------- */
		root_box_norm = init_box;            // copia (dim = n)
		inv_range.resize(n+1);               // n+1 porque después manejas espacios extendidos
		for (int j = 0; j < n; ++j)
		    inv_range[j] = 1.0 / root_box_norm[j].diam();   // diámetro > 0 por hipótesis
		// la coordenada goal_var (y) no se normaliza porque no forma parte de los centros
		/* -------------------------------------------------------------- */


		buffer.contract(loup);
		uplo = NEG_INFINITY;
		uplo_of_epsboxes = POS_INFINITY;
		nb_cells = 0;
		buffer.flush();
		Cell *root = new Cell(IntervalVector(n + 1));
		write_ext_box(init_box, root->box);
		// add data required by the bisector
		bsc.add_property(init_box, root->prop);
		// add data required by the contractor
		ctc.add_property(init_box, root->prop);
		// add data required by the buffer
		buffer.add_property(init_box, root->prop);
		// add data required by the loup finder
		loup_finder.add_property(init_box, root->prop);
		// cout << "**** Properties ****\n" << root->prop << endl;
		loup_changed = false;
		initial_loup = obj_init_bound;
		loup_point = init_box; //.set_empty();
		time = 0;
		if (cov)
			delete cov;
		cov = new CovOptimData(extended_COV ? n + 1 : n, extended_COV);
		cov->data->_optim_time = 0;
		cov->data->_optim_nb_cells = 0;
		if (trace >= 1) {
        	double initial_volume = calculate_box_volume(init_box);
        	cout << "[Optimizer START] Initial decision space volume (" << init_box.size() << " vars): " 
        	     << initial_volume << endl;
    	}

		handle_cell(*root);
	}
	void Optimizer::start(const CovOptimData &data, double obj_init_bound)
	{
		loup = obj_init_bound;
		// Just to initialize the "loup" for the buffer
		// TODO: replace with a set_loup function
		buffer.contract(loup);
		uplo = data.uplo();
		loup = data.loup();
		loup_point = data.loup_point();
		uplo_of_epsboxes = POS_INFINITY;
		nb_cells = 0;
		buffer.flush();
		for (size_t i = loup_point.is_empty() ? 0 : 1; i < data.size(); i++)
		{
			IntervalVector box(n + 1);
			if (data.is_extended_space())
				box = data[i];
			else
			{
				write_ext_box(data[i], box);
				box[goal_var] = Interval(uplo, loup);
				ctc.contract(box);
				if (box.is_empty())
					continue;
			}
			Cell *cell = new Cell(box);
			// add data required by the cell buffer
			buffer.add_property(box, cell->prop);
			// add data required by the bisector
			bsc.add_property(box, cell->prop);
			// add data required by the contractor
			ctc.add_property(box, cell->prop);
			// add data required by the loup finder
			loup_finder.add_property(box, cell->prop);
			buffer.push(cell);
		}
		loup_changed = false;
		initial_loup = obj_init_bound;
		time = 0;
		if (cov)
			delete cov;
		cov = new CovOptimData(extended_COV ? n + 1 : n, extended_COV);
		cov->data->_optim_time = data.time();
		cov->data->_optim_nb_cells = data.nb_cells();
	}
	Optimizer::Status Optimizer::optimize()
	{
		Timer timer;
		timer.start();
		update_uplo();
	
		try
		{
			//cout << "Inicio Clustering normal" << endl;
			while (!buffer.empty())
			{
				// cout << buffer.size() << endl;
				loup_changed = false;
				// for double heap , choose randomly the buffer : top has to be called before pop
				Cell *c = buffer.top();
				if (trace >= 2)
					cout << " current box " << c->box << endl;
				try
				{
					pair<Cell *, Cell *> new_cells = bsc.bisect(*c);
					buffer.pop();
					delete c;	   // deletes the cell.
					nb_cells += 2; // counting the cells handled ( in previous versions nb_cells was the number of cells put into the buffer after being handled)
					handle_cell(*new_cells.first);
					handle_cell(*new_cells.second);
					if (uplo_of_epsboxes == NEG_INFINITY)
					{
						break;
					}
					if (loup_changed)
					{
						// In case of a new upper bound (loup_changed == true), all the boxes
						// with a lower bound greater than (loup - goal_prec) are removed and deleted.
						// Note: if contraction was before bisection, we could have the problem
						// that the current cell is removed by contractHeap. See comments in
						// older version of the code (before revision 284).
						double ymax = compute_ymax();
						buffer.contract(ymax);
						// cout << " now buffer is contracted and min=" << buffer.minimum() << endl;
						// TODO: check if happens. What is the return code in this case?
						if (ymax <= NEG_INFINITY)
						{
							if (trace)
								cout << " infinite value for the minimum " << endl;
							break;
						}
					}
					update_uplo();

					// ── Control de estancamiento (iteraciones o tamaño del buffer) ─────────
					if (loup_changed)
                    {
                        stagnation_counter = 0;
                    }
                    if (loup != POS_INFINITY)
                    {
                        ++stagnation_counter;
                    }
                    
                    // Dispara reinicio si se cumple cualquiera de los dos umbrales
                    if (stagnation_counter >= restart_threshold || buffer.size() >= node_threshold)
                    {
						// ── PASO 1: DIAGNÓSTICO DE ESTADO (ANTES DEL REINICIO) ──
						double stag_ratio = (double)stagnation_counter / restart_threshold;
						double buffer_ratio = (double)buffer.size() / node_threshold;
						
						int stagnation_level;
						std::string level_name;
						
						// Clasificación en 4 niveles según umbrales absolutos
						if (restart_threshold > 2000000 || node_threshold > 120000) {
							stagnation_level = 4;  // CRÍTICO
							level_name = "CRÍTICO";
						} else if (restart_threshold >= 100000 || node_threshold >= 55000) {
							stagnation_level = 3;  // SEVERO
							level_name = "SEVERO";
						} else if (restart_threshold >= 50000 || node_threshold >= 20000) {
							stagnation_level = 2;  // MODERADO
							level_name = "MODERADO";
						} else {
							stagnation_level = 1;  // LEVE
							level_name = "LEVE";
						}
						
						// Identificar qué umbral gatilló
						bool triggered_by_iterations = (stagnation_counter >= restart_threshold);
						bool triggered_by_buffer = (buffer.size() >= node_threshold);
						
						if (trace >= 1) {
							cout << "\n┌─────────────────────────────────────────────────────────────┐\n";
							cout << "│ REINICIO MULTI-NIVEL - DIAGNÓSTICO DE ESTADO                │\n";
							cout << "├─────────────────────────────────────────────────────────────┤\n";
							cout << "│ Nivel detectado: " << level_name;
							for (size_t i = level_name.length(); i < 48; ++i) cout << " ";
							cout << "│\n";
							
							cout << "│ Disparado por: ";
							if (triggered_by_iterations) 
								cout << stagnation_counter << " iteraciones sin mejora";
							else 
								cout << "buffer.size()==" << buffer.size();
							
							int chars_printed = triggered_by_iterations ? 
								(18 + std::to_string(stagnation_counter).length() + 23) : 
								(15 + std::to_string(buffer.size()).length());
							for (int i = chars_printed; i < 58; ++i) cout << " ";
							cout << "│\n";
							
							// ── Formatear porcentajes con ostringstream para evitar contaminar cout ──
							std::ostringstream pct_stream;
							
							// Porcentaje de stagnation
							pct_stream << std::fixed << std::setprecision(1) << (100.0 * stag_ratio);
							std::string stag_pct = pct_stream.str();
							
							cout << "│ Stagnation: " << stagnation_counter << "/" << restart_threshold 
								<< " (" << stag_pct << "%)";
							
							int stag_chars = 13 + std::to_string(stagnation_counter).length() + 
											std::to_string(restart_threshold).length() + 
											stag_pct.length() + 4;
							for (int i = stag_chars; i < 58; ++i) cout << " ";
							cout << "│\n";
							
							// Porcentaje de buffer
							pct_stream.str("");
							pct_stream.clear();
							pct_stream << std::fixed << std::setprecision(1) << (100.0 * buffer_ratio);
							std::string buf_pct = pct_stream.str();
							
							cout << "│ Buffer:     " << buffer.size() << "/" << node_threshold 
								<< " (" << buf_pct << "%)";
							
							int buf_chars = 13 + std::to_string(buffer.size()).length() + 
											std::to_string(node_threshold).length() + 
											buf_pct.length() + 4;
							for (int i = buf_chars; i < 58; ++i) cout << " ";
							cout << "│\n";
							
							cout << "└─────────────────────────────────────────────────────────────┘\n";
							
							// *** IMPORTANTE: cout mantiene su formato original (fixed + precision(12)) ***
						}
						
						// ── PASO 2: AJUSTE DEL UMBRAL DE VOLUMEN SEGÚN NIVEL ──
						
						switch (stagnation_level) {
							case 1:  // LEVE
								clustering_params.hull_volume_threshold_fraction = 2.5;  // Más restrictivo
								break;
							case 2:  // MODERADO
								clustering_params.hull_volume_threshold_fraction =2.8;   // Restrictivo
								break;
							case 3:  // SEVERO
								clustering_params.hull_volume_threshold_fraction = 3.2;  // Permisivo
								break;
							case 4:  // CRÍTICO
								clustering_params.hull_volume_threshold_fraction = 3.5;  // Más permisivo
								break;
						}
						
						if (trace >= 1) {
							cout << "│ Hull volume threshold ajustado: " 
								<< clustering_params.hull_volume_threshold_fraction << "×\n";
						}
						
						// ── PASO 3: CAPTURAR ESTADO PRE-REINICIO ──
						buffer_size_before_restart = buffer.size();
						loup_before_restart = loup;
						double uplo_before = uplo; 
						
						// ── PASO 4: EJECUTAR REINICIO ──
						cluster_restart();
						stagnation_counter = 0;
						
						// ── PASO 5: EVALUAR EFECTIVIDAD DEL REINICIO ──
						size_t buffer_size_after = buffer.size();
						double loup_after = loup;
						
						bool buffer_reduced = (buffer_size_after < buffer_size_before_restart);
						bool loup_improved = (loup_after < loup_before_restart - abs_eps_f);
						bool was_successful = (buffer_reduced || loup_improved);

						// ── PASO 6: AJUSTE DINÁMICO DE UMBRALES SEGÚN NIVEL + EFECTIVIDAD ──
						
						// Función auxiliar para calcular multiplicador según tamaño
						auto get_mult = [](size_t threshold, bool is_large_scale, int level, bool successful) -> double {
							if (successful) {
								// Éxito: multiplicadores mayores
								if (is_large_scale) {
									// Para node_threshold
									if (level == 1) return 2.5;
									else if (level == 2) return 1.9;
									else if (level == 3) return 1.4;
									else return 1.15; // CRÍTICO: crecimiento mínimo
								} else {
									// Para restart_threshold
									if (level == 1) return 3.0;
									else if (level == 2) return 2.1;
									else if (level == 3) return 1.4;
									else return 1.15; // CRÍTICO: crecimiento mínimo
								}
							} else {
								// Fallo: multiplicadores menores
								if (is_large_scale) {
									// Para node_threshold
									if (level == 1) return 2.2;      // LEVE (-0.3)
									else if (level == 2) return 1.7; // MODERADO (-0.2)
									else if (level == 3) return 1.3; // SEVERO (-0.1)
								} else {
									// Para restart_threshold
									if (level == 1) return 2.7;      // LEVE (-0.3)
									else if (level == 2) return 1.9; // MODERADO (-0.2)
									else if (level == 3) return 1.3; // SEVERO (-0.1)
								}
							}
							
						};
						
						double mult_iters, mult_buffer;
						
						if (was_successful) {
							restart_stats.successful_restarts++;
							
							if (triggered_by_iterations && !triggered_by_buffer) {
								// Solo iteraciones: ese umbral crece más
								mult_iters = get_mult(restart_threshold, false, stagnation_level, true);
								mult_buffer = mult_iters * 0.8;
								
							} else if (triggered_by_buffer && !triggered_by_iterations) {
								// Solo buffer: ese umbral crece más
								mult_buffer = get_mult(node_threshold, true, stagnation_level, true);
								mult_iters = mult_buffer * 0.8;
								
							} else {
								// Ambos gatillaron: multiplicadores balanceados
								mult_iters = get_mult(restart_threshold, false, stagnation_level, true);
								mult_buffer = get_mult(node_threshold, true, stagnation_level, true);
							}
							
						}else {
							restart_stats.failed_restarts++;

							if (stagnation_level == 4){
								mult_iters = get_mult(restart_threshold, false, stagnation_level, false);
								mult_buffer = get_mult(node_threshold, true, stagnation_level, false);

							} else {
								// Fallo: multiplicadores reducidos
								if (triggered_by_iterations && !triggered_by_buffer) {
									mult_iters = get_mult(restart_threshold, false, stagnation_level, false);
									mult_buffer = mult_iters * 0.9;
									
								} else if (triggered_by_buffer && !triggered_by_iterations) {
									mult_buffer = get_mult(node_threshold, true, stagnation_level, false);
									mult_iters = mult_buffer * 0.9;
									
								} else {
									// Ambos gatillaron con fallo
									mult_iters = get_mult(restart_threshold, false, stagnation_level, false);
									mult_buffer = get_mult(node_threshold, true, stagnation_level, false);
								}
							}
						}
						
						// ── PASO 7: APLICAR MULTIPLICADORES CON LÍMITES MÁXIMOS ──
						const int MAX_RESTART = 300000;
						const size_t MAX_NODE = 150000;
						
						int new_restart = (int)(restart_threshold * mult_iters);
						size_t new_node = (size_t)(node_threshold * mult_buffer);
						
						bool restart_at_max = false;
						bool node_at_max = false;
						
						if (restart_threshold >= MAX_RESTART) {
							new_restart = MAX_RESTART;
							restart_at_max = true;
						} else {
							new_restart = std::min(new_restart, MAX_RESTART);
							if (new_restart == MAX_RESTART) restart_at_max = true;
						}
						
						if (node_threshold >= MAX_NODE) {
							new_node = MAX_NODE;
							node_at_max = true;
						} else {
							new_node = std::min(new_node, MAX_NODE);
							if (new_node == MAX_NODE) node_at_max = true;
						}
						
						restart_threshold = new_restart;
						node_threshold = new_node;
						
						// ── PASO 8: REPORTE DETALLADO ──
						if (trace >= 1) {
							cout << "\n┌─────────────────────────────────────────────────────────────┐\n";
							cout << "│ RESULTADO DEL REINICIO                                      │\n";
							cout << "├─────────────────────────────────────────────────────────────┤\n";
							cout << "│ Estado: " << (was_successful ? "✓ EXITOSO" : "✗ FALLIDO");
							for (int i = (was_successful ? 9 : 9); i < 58; ++i) cout << " ";
							cout << "│\n";
							
							cout << "│ Buffer: " << buffer_size_before_restart << " → " 
								<< buffer_size_after;
							if (buffer_reduced) cout << " ✓";
							
							int buf_line_len = 9 + std::to_string(buffer_size_before_restart).length() + 
											std::to_string(buffer_size_after).length() + 3;
							if (buffer_reduced) buf_line_len += 2;
							for (int i = buf_line_len; i < 58; ++i) cout << " ";
							cout << "│\n";
							
							if (loup_improved) {
								cout << "│ Loup:   " << loup_before_restart << " → " 
									<< loup_after << " ✓";
								for (int i = 0; i < 30; ++i) cout << " ";
								cout << "│\n";
							}
							
							cout << "│ Nivel:  " << stagnation_level << " (" << level_name << ")";
							for (size_t i = level_name.length(); i < 43; ++i) cout << " ";
							cout << "│\n";
							
							// Preparar formateo de multiplicadores
							std::ostringstream mult_stream;

							// Formatear mult_iters
							mult_stream << std::fixed << std::setprecision(1) << mult_iters;
							std::string mult_iters_str = mult_stream.str();

							// Formatear mult_buffer (reutilizar stream)
							mult_stream.str("");  // Limpiar contenido
							mult_stream.clear();  // Limpiar flags de error
							mult_stream << std::fixed << std::setprecision(1) << mult_buffer; // Formatear
							std::string mult_buffer_str = mult_stream.str(); // Obtener string

							cout << "│ Multiplicadores: iters ×" << mult_iters_str
								<< ", buffer ×" << mult_buffer_str;
							for (int i = 0; i < 23; ++i) cout << " ";
							cout << "│\n";
							
							cout << "│ Nuevos umbrales:";
							for (int i = 0; i < 42; ++i) cout << " ";
							cout << "│\n";
							
							cout << "│   restart_threshold = " << restart_threshold;
							if (restart_at_max) cout << " [MAX]";
							for (size_t i = 23 + std::to_string(restart_threshold).length() + 
											(restart_at_max ? 6 : 0); i < 58; ++i) cout << " ";
							cout << "│\n";
							
							cout << "│   node_threshold    = " << node_threshold;
							if (node_at_max) cout << " [MAX]";
							for (size_t i = 23 + std::to_string(node_threshold).length() + 
											(node_at_max ? 6 : 0); i < 58; ++i) cout << " ";
							cout << "│\n";
							
							cout << "│ Hull threshold: " << clustering_params.hull_volume_threshold_fraction 
								<< "×";
							for (int i = 0; i < 38; ++i) cout << " ";
							cout << "│\n";
							
							cout << "└─────────────────────────────────────────────────────────────┘\n\n";

						}
						
						continue; // volvemos a la cabecera del while con nuevas cajas
					}

					// ───────────────────────────────────────────────────────────────────────
					if (!anticipated_upper_bounding) // useless to check precision on objective if 'true'
						if (get_obj_rel_prec() < rel_eps_f || get_obj_abs_prec() < abs_eps_f)
							break;
					if (timeout > 0)
						timer.check(timeout); // TODO: not reentrant, JN: done
					time = timer.get_time();
				}
				catch (NoBisectableVariableException &)
				{
					update_uplo_of_epsboxes((c->box)[goal_var].lb());
					buffer.pop();
					delete c;	   // deletes the cell.
					update_uplo(); // the heap has changed -> recalculate the uplo (eg: if not in best-first search)
				}
			}

			timer.stop();
			time = timer.get_time();
			// No solution found and optimization stopped with empty buffer
			// before the required precision is reached => means infeasible problem
			if (uplo_of_epsboxes == NEG_INFINITY)
				status = UNBOUNDED_OBJ;
			else if (uplo_of_epsboxes == POS_INFINITY && (loup == POS_INFINITY || (loup == initial_loup && abs_eps_f == 0 && rel_eps_f == 0)))
				status = INFEASIBLE;
			else if (loup == initial_loup)
				status = NO_FEASIBLE_FOUND;
			else if (get_obj_rel_prec() > rel_eps_f && get_obj_abs_prec() > abs_eps_f)
				status = UNREACHED_PREC;
			else
				status = SUCCESS;


			//STATS NUEVAS
			if (trace >= 0) {
                cout << endl;
                cout << "------------------------------------------" << endl;
                cout << " RESUMEN DE ESTADISTICAS DE CLUSTERING" << endl;
                cout << "------------------------------------------" << endl;
                cout << " Reinicios totales gatillados:  " << restart_stats.total_restarts_triggered << endl;
                cout << " Clústeres totales formados:    " << restart_stats.total_clusters_formed << endl;
                cout << " Hulls finales creados:         " << restart_stats.total_hulls_created << endl;
                cout << " Nodos totales unidos en hulls: " << restart_stats.total_nodes_merged << endl;
                
                if (restart_stats.total_restarts_triggered > 0) {
                    cout << "   Reinicios exitosos:          " << restart_stats.successful_restarts << endl;
                    cout << "   Reinicios fallidos:          " << restart_stats.failed_restarts << endl;
                }
                if (restart_stats.total_hulls_created > 0) {
                    double avg_nodes_per_hull = (double)restart_stats.total_nodes_merged / 
                                               restart_stats.total_hulls_created;
                    cout << " Promedio de nodos/hull:         " << std::fixed << std::setprecision(2) 
                         << avg_nodes_per_hull << endl;
                }
                cout << "------------------------------------------" << endl << endl;
            }	
			//STATS NUEVAS
		}
		catch (TimeOutException &)
		{
			status = TIME_OUT;
		}
		/* TODO: cannot retrieve variable names here. */
		for (int i = 0; i < (extended_COV ? n + 1 : n); i++)
			cov->data->_optim_var_names.push_back(string(""));
		cov->data->_optim_optimizer_status = (unsigned int)status;
		cov->data->_optim_uplo = uplo;
		cov->data->_optim_uplo_of_epsboxes = uplo_of_epsboxes;
		cov->data->_optim_loup = loup;
		cov->data->_optim_time += time;
		cov->data->_optim_nb_cells += nb_cells;
		cov->data->_optim_loup_point = loup_point;
		// for conversion between original/extended boxes
		IntervalVector tmp(extended_COV ? n + 1 : n);
		// by convention, the first box has to be the loup-point.
		if (extended_COV)
		{
			write_ext_box(loup_point, tmp);
			tmp[goal_var] = Interval(uplo, loup);
			cov->add(tmp);
		}
		else
		{
			cov->add(loup_point);
		}
		while (!buffer.empty())
		{
			Cell *cell = buffer.top();
			if (extended_COV)
				cov->add(cell->box);
			else
			{
				read_ext_box(cell->box, tmp);
				cov->add(tmp);
			}
			delete buffer.pop();
		}
		return status;
	}
	namespace
	{
		const char *green()
		{
#ifndef _WIN32
			return "\033[32m";
#else
			return "";
#endif
		}
		const char *red()
		{
#ifndef _WIN32
			return "\033[31m";
#else
			return "";
#endif
		}
		const char *white()
		{
#ifndef _WIN32
			return "\033[0m";
#else
			return "";
#endif
		}
	}
	void Optimizer::report()
	{
		if (!cov || !buffer.empty())
		{ // not started
			cout << " not started." << endl;
			return;
		}
		switch (status)
		{
		case SUCCESS:
			cout << green() << " optimization successful!" << endl;
			break;
		case INFEASIBLE:
			cout << red() << " infeasible problem" << endl;
			break;
		case NO_FEASIBLE_FOUND:
			cout << red() << " no feasible point found (the problem may be infeasible)" << endl;
			break;
		case UNBOUNDED_OBJ:
			cout << red() << " possibly unbounded objective (f*=-oo)" << endl;
			break;
		case TIME_OUT:
			cout << red() << " time limit " << timeout << "s. reached " << endl;
			break;
		case UNREACHED_PREC:
			cout << red() << " unreached precision" << endl;
			break;
		}
		cout << white() << endl;
		// No solution found and optimization stopped with empty buffer
		// before the required precision is reached => means infeasible problem
		if (status == INFEASIBLE)
		{
			cout << " infeasible problem " << endl;
		}
		else
		{
			cout << " f* in\t[" << uplo << "," << loup << "]" << endl;
			cout << "\t(best bound)" << endl
				 << endl;
			if (loup == initial_loup)
				cout << " x* =\t--\n\t(no feasible point found)" << endl;
			else
			{
				if (loup_finder.rigorous())
					cout << " x* in\t" << loup_point << endl;
				else
					cout << " x* =\t" << loup_point.lb() << endl;
				cout << "\t(best feasible point)" << endl;
			}
			cout << endl;
			double rel_prec = get_obj_rel_prec();
			double abs_prec = get_obj_abs_prec();
			cout << " relative precision on f*:\t" << rel_prec;
			if (rel_prec <= rel_eps_f)
				cout << green() << " [passed] " << white();
			cout << endl;
			cout << " absolute precision on f*:\t" << abs_prec;
			if (abs_prec <= abs_eps_f)
				cout << green() << " [passed] " << white();
			cout << endl;
		}
		cout << " cpu time used:\t\t\t" << time << "s";
		if (cov->time() != time)
			cout << " [total=" << cov->time() << "]";
		cout << endl;
		cout << " number of cells:\t\t" << nb_cells;
		if (cov->nb_cells() != nb_cells)
			cout << " [total=" << cov->nb_cells() << "]";
		cout << endl
			 << endl;
		if (statistics)
			cout << " ===== Statistics ====" << endl
				 << endl
				 << *statistics << endl;
	}

	
    void Optimizer::cluster_restart()
    {
		restart_stats.total_restarts_triggered++; // <--- CONTADOR DE REINICIOS
        if (trace)
            cout << "[cluster_restart] Iniciando reinicio por clustering (";
			switch (clustering_params.choice) {
				case ClusteringParams::Algorithm::KMEANS_PLUS:
					cout << "K-Means++";
					break;
				case ClusteringParams::Algorithm::HDBSCAN:
                    cout << "HDBSCAN";
                    break;
			}
            cout << ")...\n";

        // 1) Sacar TODAS las celdas del buffer
        std::vector<Cell *> active_cells;
        while (!buffer.empty())
            active_cells.push_back(buffer.pop());

        size_t N = active_cells.size();
        if (N == 0)
        {
            if (trace)
                cout << "[cluster_restart] buffer vacío, nada que hacer.\n";
            return;
        }

        if (trace)
            cout << "[cluster_restart] celdas extraídas: " << N << "\n";

		// calcular volumen
        double sum_of_original_volumes = 0.0;
        bool any_original_volume_is_infinite = false;
        for (Cell* cell_ptr : active_cells) {
            if (cell_ptr) { // Comprobación por si acaso
                double vol = calculate_box_volume(cell_ptr->box); // Usando tu función calculate_box_volume
                if (vol == POS_INFINITY) { // POS_INFINITY está definido en <cmath> o <limits>
                    any_original_volume_is_infinite = true;
                    // Si un volumen es infinito, la suma total será infinita
                }
                if (!any_original_volume_is_infinite) {
                    sum_of_original_volumes += vol;
                } else {
                    sum_of_original_volumes = POS_INFINITY; // Marcar la suma como infinita
                }
            }
        }

        if (trace >= 1) {
            if (any_original_volume_is_infinite) {
                cout << "[cluster_restart] Suma de volúmenes PRE-CLUSTERING (celdas originales): POS_INFINITY" << endl;
            } else {
                cout << "[cluster_restart] Suma de volúmenes PRE-CLUSTERING (celdas originales): " << sum_of_original_volumes << endl;
            }
		}

        // **** FIN DEL CÁLCULO DE VOLUMEN PRE-CLUSTERING ****

        const int dim = n + 1;

        // 2) Calcular centros
        std::vector<Point> centers;
        centers.reserve(N);
        for (Cell *c_ptr : active_cells)
        {
            const IntervalVector &box = c_ptr->box;
            Point p(dim);
            for (int j = 0; j < dim; ++j)
        		if (clustering_params.use_normalization) {
        		    if (j == goal_var) {
        		        p[j] = box[j].mid();
        		    } else {
        		        double mid = box[j].mid();
        		        double lb0 = root_box_norm[j].lb();
        		        p[j] = (mid - lb0) * inv_range[j]; // CÓDIGO DE NORMALIZACIÓN
        		    }
        		}
        		// Si el interruptor está apagado
        		else {
        		    p[j] = box[j].mid(); // CÓDIGO SIN NORMALIZACIÓN (usa el centro directo)
        		}
            centers.push_back(std::move(p));
        }



        // 3) Ejecutar Clustering 
        std::vector<int> result_labels;
        int actual_num_clusters = 0;

        // Variables para el log de VOLUMEN POST-CLUSTERING 
        double sum_of_hulls_volume_created = 0.0;
        int num_hulls_actually_formed = 0;
        bool any_formed_hull_volume_is_infinite = false;
        
        if (clustering_params.choice == ClusteringParams::Algorithm::KMEANS_PLUS) //USAR K-MEANS++
        {
            int k_for_kmeans = (N > 0) ? std::max(1, (int)std::sqrt((double)N / 2.0)) : 0;

            if (N > 0 && (size_t)k_for_kmeans > N) {
                if (trace) cout << "[cluster_restart] K-Means++: k (" << k_for_kmeans << ") > N (" << N << "). Usando k=N.\n";
                k_for_kmeans = N;
            }

            if (N == 0) {
                actual_num_clusters = 0;
            } else if (k_for_kmeans == 0 && N > 0) {
                if (trace) cout << "[cluster_restart] K-Means++: k=0 con N>0 puntos. Agrupando todo en un clúster.\n";
                result_labels.assign(N,0);
                actual_num_clusters = 1;
            } else if (N > 0) {
                ClusterResult kmeans_res = kmeans_plus(centers, k_for_kmeans);  // ← K-Means++
                result_labels = kmeans_res.labels;
                actual_num_clusters = kmeans_res.n_clusters;
            } else {
                actual_num_clusters = 0;
            }
            
            if (trace) {
                cout << "[cluster_restart] K-Means++ -> " << actual_num_clusters
                     << " clústeres (k solicitado=" << k_for_kmeans << ").\n";
            }
        }
		else if (clustering_params.choice == ClusteringParams::Algorithm::HDBSCAN)
        {
            // HDBSCAN: Clustering jerárquico sin parámetro eps
            int min_cluster_size = clustering_params.hdbscan_min_cluster_size;
            int min_samples = clustering_params.hdbscan_min_samples;

			// Ajuste dinámico: si N > 1000, usar √N
			if (N > 1000) {
				min_cluster_size = std::max(15, (int)std::sqrt((double)N));
				if (trace) {
					cout << "[cluster_restart] HDBSCAN: min_cluster_size ajustado a √N = " 
						<< min_cluster_size << " (N=" << N << ")\n";
				}
			}
            
            if (N > 0 && N < (size_t)min_cluster_size) {
                if (trace) {
                    cout << "[cluster_restart] HDBSCAN: No hay suficientes puntos (" << N
                         << ") para min_cluster_size=" << min_cluster_size
                         << ". Marcando todos como ruido.\n";
                }
                result_labels.assign(N, -1);  // Todos ruido
                actual_num_clusters = 0;
            }
            else if (N == 0) {
                actual_num_clusters = 0;
            }
            else {
                HdbscanClusterResult hdbscan_res = hdbscan(
                    centers, 
                    min_cluster_size, 
                    min_samples
                );
                result_labels = hdbscan_res.labels;
                actual_num_clusters = hdbscan_res.num_clusters;
            }
            
            if (trace) {
                cout << "[cluster_restart] HDBSCAN -> " << actual_num_clusters 
                     << " clústeres encontrados (min_cluster_size=" << min_cluster_size;
                if (min_samples != -1)
                    cout << ", min_samples=" << min_samples;
                cout << ").\n";
                
                if (N > 0) {
                    int noise_count = 0;
                    for (size_t i = 0; i < N; ++i)
                        if (result_labels[i] == -1)
                            noise_count++;
                    if (noise_count > 0) {
                        cout << " HDBSCAN Ruido: " << noise_count << " de " << N 
                             << " nodos (" << (double)noise_count / N * 100 << "%)\n";
                    }
                }
            }
        }
        
		restart_stats.total_clusters_formed += actual_num_clusters;
     	// --- PROCESAMIENTO DE CLÚSTERES Y CREACIÓN DE HULLS  ---
    	// 1) Agrupamos punteros a celdas por etiqueta de clúster
    	std::vector<std::vector<Cell*>> clusters_members(actual_num_clusters);
    	std::vector<Cell*> noise_cells;
    	for (size_t i = 0; i < N; ++i) {
    	    Cell* c = active_cells[i];
    	    int lbl = result_labels[i];
    	    // Ruido o etiqueta inválida
    	    if (lbl < 0 || lbl >= actual_num_clusters || (clustering_params.choice == ClusteringParams::Algorithm::HDBSCAN && lbl == -1)) 
			{
				noise_cells.push_back(c);
    	    } else {
    	        clusters_members[lbl].push_back(c);
    	    }
    	}
    	active_cells.clear();

    	// 2) Para cada clúster, calculamos su hull y decidimos si usarlo
    	for (int c_id = 0; c_id < actual_num_clusters; ++c_id) {
    	    auto &members = clusters_members[c_id];
    	    if (members.empty()) continue;

			double sum_cluster_volumes = 0.0;
			for (Cell* ptr : members) {
			    sum_cluster_volumes += calculate_box_volume(ptr->box);
			}

			// Umbral para este clúster (fracción definida en ClusteringParams)
			double cluster_threshold = clustering_params.hull_volume_threshold_fraction * sum_cluster_volumes;

			if (trace >= 1) {
			    cout << "[cluster_restart] Clúster " << c_id 
			         << ": sum_cluster_volumes=" << sum_cluster_volumes 
			         << ", cluster_threshold=" << cluster_threshold 
			         << endl;
			}

    	    // 2a) Construcción del hull envolvente
    	    IntervalVector hull_box(dim);
    	    hull_box.set_empty();
    	    for (Cell* ptr : members) {
    	        const auto &b = ptr->box;
    	        if (hull_box.is_empty()) {
    	            hull_box = b;
    	        } else {
    	            for (int j = 0; j < dim; ++j)
    	                hull_box[j] |= b[j];
    	        }
    	    }

    	    // 2b) Cálculo de volumen del hull
    	    double hull_vol = calculate_box_volume(hull_box);

			if (trace >= 1) {
    		    cout << "[cluster_restart] Clúster " << c_id
    		         << ": hull_vol=" << hull_vol
    		         << ", threshold=" << cluster_threshold
    		         << endl;
    		}

    	    // 2c) Filtro: si supera el umbral, reinsertamos las cajas originales
    	    if (hull_vol > cluster_threshold) {
        		// — LOG DE TRACE: volumen excesivo, reinsertando celdas —
    			if (trace >= 1) {
    			    cout << "[cluster_restart] Clúster " << c_id
    			         << ": hull_vol=" << hull_vol
    			         << " > cluster_threshold=" << cluster_threshold
    			         << ". Reinsertando " << members.size() << " cajas." << endl;
    			}
    	        for (Cell* ptr : members) {
    	            buffer.add_property(ptr->box, ptr->prop);
    	            bsc          .add_property(ptr->box, ptr->prop);
    	            ctc          .add_property(ptr->box, ptr->prop);
    	            loup_finder  .add_property(ptr->box, ptr->prop);
    	            buffer.push(ptr);
    	        }
    	    }
    	    // 2d) Si está por debajo, insertamos un único hull y eliminamos las cajas
    	    else {
				if (trace >= 1) {
    			    cout << "[cluster_restart] Clúster " << c_id
    			         << ": hull_vol=" << hull_vol
    			         << " ≤ cluster_threshold=" << cluster_threshold
    			         << ". Creando hull único." << endl;
    			}

				restart_stats.total_hulls_created++;
				restart_stats.total_nodes_merged += members.size();

    	        Cell* nc = new Cell(hull_box);
    	        buffer.add_property(nc->box, nc->prop);
    	        bsc         .add_property(nc->box, nc->prop);
    	        ctc         .add_property(nc->box, nc->prop);
    	        loup_finder .add_property(nc->box, nc->prop);
    	        buffer.push(nc);
				num_hulls_actually_formed++;
				if (hull_vol == POS_INFINITY)
				    any_formed_hull_volume_is_infinite = true;
				else
				    sum_of_hulls_volume_created += hull_vol;
    	        for (Cell* ptr : members) delete ptr;
    	    }

    	    members.clear();
		}

    	// 3) Reinsertamos todas las celdas clasificadas como “ruido”
		if (trace >= 1 && !noise_cells.empty()) {
        cout << "[cluster_restart] Reinsertando ruido: "
             << noise_cells.size()
             << " celdas." << endl;
    	}
    	for (Cell* ptr : noise_cells) {
    	    buffer.add_property(ptr->box, ptr->prop);
    	    bsc         .add_property(ptr->box, ptr->prop);
    	    ctc         .add_property(ptr->box, ptr->prop);
    	    loup_finder .add_property(ptr->box, ptr->prop);
    	    buffer.push(ptr);
    	}
    	noise_cells.clear();

        // ---- Log de la SUMA de volúmenes de hulls POST-CLUSTERING ----
        if (trace >= 1) {
            if (num_hulls_actually_formed > 0) {
                if (any_formed_hull_volume_is_infinite) {
                    cout << "[cluster_restart] Suma de volúmenes POST-CLUSTERING (" << num_hulls_actually_formed 
                              << " hulls formados): POS_INFINITY." << endl;
                } else {
                    cout << "[cluster_restart] Suma de volúmenes POST-CLUSTERING (" << num_hulls_actually_formed 
                              << " hulls formados): " << sum_of_hulls_volume_created << endl;
                }
            } else if (N > 0) { 
                cout << "[cluster_restart] No se formaron hulls POST-CLUSTERING (0 clústeres o todos vacíos)." << endl;
            }
        }
        // ---- Fin log suma de volúmenes ----

        if (trace >=1 )
        {
            cout << "[cluster_restart] Completado. Buffer ahora tiene " << buffer.size() << " celdas.\n";
        }
    } // Fin de Optimizer::cluster_restart

} // end namespace ibex