#!/bin/bash
# Run Python scripts in the background without -it flag
docker exec yuasa-server python3 inference_main.py /workspace/data/OK_data1 /workspace/data/results/OK_data1 0 &
PID1=$!
docker exec yuasa-server python3 inference_main.py /workspace/data/OK_data2 /workspace/data/results/OK_data2 0 &
PID2=$!
docker exec yuasa-server python3 inference_main.py /workspace/data/OK_data3 /workspace/data/results/OK_data3 0 &
PID3=$!
docker exec yuasa-server python3 inference_main.py /workspace/data/OK_data4 /workspace/data/results/OK_data4 1 &
PID4=$!
docker exec yuasa-server python3 inference_main.py /workspace/data/OK_data5 /workspace/data/results/OK_data5 1 &
PID5=$!
docker exec yuasa-server python3 inference_main.py /workspace/data/OK_data6 /workspace/data/results/OK_data6 1 &
PID6=$!
docker exec yuasa-server python3 inference_main.py /workspace/data/OK_data7 /workspace/data/results/OK_data7 2 &
PID7=$!
docker exec yuasa-server python3 inference_main.py /workspace/data/OK_data8 /workspace/data/results/OK_data8 2 &
PID8=$!
docker exec yuasa-server python3 inference_main.py /workspace/data/OK_data9 /workspace/data/results/OK_data9 2 &
PID9=$!
docker exec yuasa-server python3 inference_main.py /workspace/data/OK_data10 /workspace/data/results/OK_data10 3 &
PID10=$!
docker exec yuasa-server python3 inference_main.py /workspace/data/OK_data11 /workspace/data/results/OK_data11 3 &
PID11=$!
docker exec yuasa-server python3 inference_main.py /workspace/data/OK_data12 /workspace/data/results/OK_data12 3 &
PID12=$!
docker exec yuasa-server python3 inference_main.py /workspace/data/OK_data13 /workspace/data/results/OK_data13 4 &
PID13=$!
docker exec yuasa-server python3 inference_main.py /workspace/data/OK_data14 /workspace/data/results/OK_data14 4 &
PID14=$!
docker exec yuasa-server python3 inference_main.py /workspace/data/OK_data15 /workspace/data/results/OK_data15 5 &
PID15=$!
docker exec yuasa-server python3 inference_main.py /workspace/data/OK_data16 /workspace/data/results/OK_data16 5 &
PID16=$!
docker exec yuasa-server python3 inference_main.py /workspace/data/OK_data17 /workspace/data/results/OK_data17 6 &
PID17=$!
docker exec yuasa-server python3 inference_main.py /workspace/data/OK_data18 /workspace/data/results/OK_data18 6 &
PID18=$!
docker exec yuasa-server python3 inference_main.py /workspace/data/OK_data19 /workspace/data/results/OK_data19 7 &
PID19=$!
docker exec yuasa-server python3 inference_main.py /workspace/data/OK_data20 /workspace/data/results/OK_data20 7 &
PID20=$!

# Wait for all processes to complete
echo "Running all Python scripts..."
wait $PID1 $PID2 $PID3 $PID4 $PID5 $PID6 $PID7 $PID8 $PID9 $PID10 $PID11 $PID12 $PID13
echo "All Python scripts have completed."
exit 0