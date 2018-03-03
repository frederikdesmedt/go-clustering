package clustering

import "errors"

// Cluster is a value expressing a single cluster.
type Cluster int

// SimpleFlatClusterer assigns a vector to exactly one cluster.
type SimpleFlatClusterer interface {
	// FindCluster returns the uniue cluster a vector is a part of.
	FindCluster(v Vector) Cluster
	// Clusters returns all the clusters this clusterer contains.
	Clusters() []Cluster
	// ClusteredPartition will split the dataset according to the cluster each element belongs to
	// such that every element in the dataset is assigned to exactly one cluster and the
	// union of all vector slices is equal to the datapoint slice of the original dataset.
	ClusteredPartition(dataset *Dataset) (map[Cluster][]Vector, error)
}

// CentroidClusterer is a clusterer that assigns a vector to a cluster such that the centroid of that cluster is at least as close to the supplied vector as every other centroid.
type CentroidClusterer []Vector

// Centroids will return a map from one of the clusters to the centroid of that cluster.
func (clusterer *CentroidClusterer) Centroids() map[Cluster]Vector {
	mapping := make(map[Cluster]Vector)
	for cluster, centroid := range []Vector(*clusterer) {
		mapping[Cluster(cluster)] = centroid
	}
	return mapping
}

// Clusters returns all the clusters this clusterer contains.
func (clusterer *CentroidClusterer) Clusters() []Cluster {
	length := len([]Vector(*clusterer))
	clusters := make([]Cluster, length)
	for i := 0; i < length; i++ {
		clusters[i] = Cluster(i)
	}
	return clusters
}

// FindCluster returns the unique cluster a vector is a part of.
func (clusterer *CentroidClusterer) FindCluster(v Vector) (Cluster, error) {
	if len(clusterer.Centroids()) == 0 {
		return -1, errors.New("There are no centroids in the CentroidClusterer")
	}
	assignedCluster, assignedClusterCentroid := Cluster(0), clusterer.Centroids()[0]
	for cluster, centroid := range clusterer.Centroids() {
		if centroid.DistanceTo(v) < assignedClusterCentroid.DistanceTo(v) {
			assignedCluster = Cluster(cluster)
			assignedClusterCentroid = centroid
		}
	}
	return assignedCluster, nil
}

// ClusteredPartition will split the dataset according to the cluster each element belongs to
// such that every element in the dataset is assigned to exactly one cluster and the
// union of all vector slices is equal to the datapoint slice of the original dataset.
func (clusterer CentroidClusterer) ClusteredPartition(dataset *Dataset) (map[Cluster][]Vector, error) {
	partition := make(map[Cluster][]Vector)
	for _, vec := range dataset.AsSlice() {
		cluster, err := clusterer.FindCluster(vec)
		if err != nil {
			return nil, err
		}
		partition[cluster] = append(partition[cluster], vec)
	}
	return partition, nil
}
