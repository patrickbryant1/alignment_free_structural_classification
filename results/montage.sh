#!/usr/bin/env bash
montage Class_bar.png Architecture_bar.png Topology_bar.png H-group_bar.png -tile 2x2 -geometry +2+2 bar.png
montage Class_hist.png Architecture_hist.png Topology_hist.png H-group_hist.png -tile 2x2 -geometry +2+2 hist.png
