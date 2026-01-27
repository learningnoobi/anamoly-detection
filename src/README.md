## Quick Start (For Testing)

If you want to test the pipeline immediately without downloading the full dataset:
- The code includes a synthetic data generator for demonstration purposes
- Run `python src/load_data.py` to generate sample data

## Dataset Structure

The dataset contains the following key features:
- Source/Destination IPs and Ports
- Protocol information
- Flow duration and packet counts
- Byte counts (forward/backward)
- Flag counts
- Label (0 = Normal, 1-9 = Various attack types)
- Type (attack category)

## Citation

```
@article{moustafa2020ton,
  title={A new distributed architecture for evaluating AI-based security systems at the edge: Network TON\_IoT datasets},
  author={Moustafa, Nour},
  journal={Sustainable Cities and Society},
  volume={72},
  pages={102994},
  year={2020},
  publisher={Elsevier}
}
```