import sys
from modulesNLM.utils import SummaryPartitions

pathPartitions = sys.argv[1]
partitions = int(sys.argv[2])

summary = SummaryPartitions.summaryPartitions(pathPartitions, partitions)
summary.getLengthPartitions()
summary.createDataSetWithPartitionID()
summary.getMeasuresCluster()
summary.createJSONSummary()
