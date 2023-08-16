import numpy as np

def normalize(x, inplace=False):
    # Exploding pixel at edge of FOV we need to ignore...
    #mask = np.zeros_like(x)
    #mask[:, 50:201, 50:201] = 1
    #x = mask*x
    if inplace:
        x -= x.min()
        x /= x.max()
    else:
        x = x - x.min()
        x = x / x.max()
    return x

class ComputeImageMetrics:

    def __init__(self, ROIs_a, ROIs_b, emissions_a, emissions_b, names_a, names_b):

        self.emissions_a = emissions_a
        self.emissions_b = emissions_b
        
        # Only use voxels where mask value is 1
        def _threshold_ROI_mask(ROIs_mask):

            for i in range(len(ROIs_mask)):
                ROIs_mask[i][ROIs_mask[i] < 1] = 0
            return ROIs_mask
            
        self.ROIs_a = _threshold_ROI_mask(ROIs_a)
        self.ROIs_b = _threshold_ROI_mask(ROIs_b)
        self.names_a = names_a
        self.names_b = names_b
    
    def _compute_std(self, x):
            # STANDARD DEVIATION
            # abar = ROI average uptake
            # Ka = number of ROIs
            # bbar = background average uptake
            # Kb = number of background ROIs
            # CRC = 1/R \sum_{r=1}^{R} (abar/bbar - 1)/(atrue/btrue - 1)
            STDval = []
            for i in range(len(self.ROIs_b)):
                STDval.append(np.std(x[np.nonzero(
                    self.ROIs_b[i]
                    )]))
            return STDval

    def _compute_crc(self, x):
            # CONTRAST RECOVERY COEFFICIENT
            # abar = ROI average uptake
            # Ka = number of ROIs
            # bbar = background average uptake
            # Kb = number of background ROIs
            # CRC = 1/R \sum_{r=1}^{R} (abar/bbar - 1)/(atrue/btrue - 1)
            CRCval = []
            for i in range(len(self.ROIs_a)):
                abar = np.mean(
                    x[np.nonzero(self.ROIs_a[i]
                    )]
                )
                bbar = np.mean(x[np.nonzero(
                    self.ROIs_b[i]
                    )]
                )
                atrue = self.emissions_a[i]
                btrue = self.emissions_b[i]
                CRCval.append((abar / bbar - 1) / (atrue / btrue - 1))
            return CRCval

    def get_all_metrics(self, x):

        return self._compute_crc(x), self._compute_std(x)
