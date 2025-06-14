import torch
import numpy as np
import os
"""
Here you define the graph for a circuit
"""

class GraphAMPNMCF:
    """                                                                                                                           

    node 0 : M0 , node 1 : M1 , node 2 : M2 , node 3 : M3 , node 4 : M4 , node 5 : M5
    node 6 : M6 , node 7 : M7 , node 8 : M8 , node 9 : M9 , node 10 : M10 , node 11 : M11
    node 12 : M12 , node 13 : M13 , node 14 : M14 , node 15 : M15 , node 16 : M16 , node17 : M17 ,
    node 18 : M18 , node 19 : M19 , node 20 : M20 , node 21 : M21 , node 22 : M22 ,   
    node23 : M23 , node24 : Ib , node25 : VDD , node26 : GND , node27 : C0 , node28 : C1

    """
    def __init__(self):        
        # self.device = torch.device(
        #     "cuda:0" if torch.cuda.is_available() else "cpu"
        # )
        
        self.device = torch.device(
           "cpu"
        )
        self.ckt_hierarchy = (
                      ('M0','x1.XM0','pfet_01v8','m'),
                      ('M1','x1.XM1','pfet_01v8','m'),
                      ('M2','x1.XM2','pfet_01v8','m'),
                      ('M3','x1.XM3','pfet_01v8','m'),
                      ('M4','x1.XM4','pfet_01v8','m'),
                      ('M5','x1.XM5','pfet_01v8','m'),
                      ('M6','x1.XM6','pfet_01v8','m'),
                      ('M7','x1.XM7','pfet_01v8','m'),
                      ('M8','x1.XM8','pfet_01v8','m'),
                      ('M9','x1.XM9','pfet_01v8','m'),
                      ('M10','x1.XM10','pfet_01v8','m'),
                      ('M11','x1.XM11','pfet_01v8','m'),
                      ('M12','x1.XM12','nfet_01v8','m'),
                      ('M13','x1.XM13','nfet_01v8','m'),
                      ('M14','x1.XM14','nfet_01v8','m'),
                      ('M15','x1.XM15','nfet_01v8','m'),
                      ('M16','x1.XM16','nfet_01v8','m'),
                      ('M17','x1.XM17','nfet_01v8','m'),
                      ('M18','x1.XM18','nfet_01v8','m'),
                      ('M19','x1.XM19','nfet_01v8','m'),
                      ('M20','x1.XM20','nfet_01v8','m'),
                      ('M21','x1.XM21','nfet_01v8','m'),
                      ('M22','x1.XM22','nfet_01v8','m'),
                      ('M23','x1.XM23','nfet_01v8','m'),

                      ('Ib','','Ib','i'),
                      ('C0','x1.XC0','cap_mim_m3_1','c'),
                      ('C1','x1.XC1','cap_mim_m3_1','c')
                     )    

        self.op = {'M0':{},'M1':{},'M2':{},'M3':{},'M4':{},'M5':{},'M6':{},'M7':{},'M8':{}, 'M9':{},'M10':{},'M10':{},'M11':{},
                'M12':{},'M13':{},'M14':{},'M15':{},'M16':{},'M17':{},'M18':{},'M19':{},'M20':{},'M21':{},'M22':{},'M23':{},
                'Ib':{},'C0':{},'C1':{}
                 }

        self.edge_index = torch.tensor([
          [0,1], [1,0], [0,2], [2,0], [0,3], [3,0], [0,4], [4,0], [0,7], [7,0], [0,24], [24,0], [0,25], [25,0], 
          [1,2], [2,1], [1,3], [3,1], [1,4], [4,1], [1,7], [7,1], [1,25], [25,1], [1,12], [12,1], [1,24], [24,1], [1,17], [17,1], [1,18], [18,1], [1,19], [19,1], [1,20], [20,1], 
          [2,3], [3,2], [2,4], [4,2], [2,7], [7,2], [2,24], [24,2], [2,13], [13,2], [2,25], [25,2],  
          [3,4], [4,3], [3,7], [7,3], [3,24], [24,3], [3,25], [25,3], [3,12], [12,3], [3,13], [13,3], [3,14], [14,3], [3,15], [15,3], [3,16], [16,3], 
          [4,7], [7,4], [4,24], [24,4], [4,8], [8,4], [4,9], [9,4], [4,25], [25,4],
          [5,6], [6,5], [5,15], [15,5], [5,25], [25,5], 
          [6,15], [15,6], [6,16], [16,6], [6,10], [10,6], [6,11], [11,6], [6,25], [25,6], [6,27], [27,6], 
          [7,24], [24,7], [7,25], [25,7], [7,22], [22,7], [7,23], [23,7], [7,28], [28,7],
          [8,9], [9,8], [8,15], [15,8], [8,19], [19,8], 
          [9,16], [16,9], [9,20], [20,9], 
          [10,11], [11,10], [10,16], [16,10], [10,21], [21,10], [10,22], [22,10], [10,25], [25,10], [10,27], [27,10],
          [11,25], [25,11], [11,16], [16,11], [11,23], [23,11], [11,27], [27,11], [11,28], [28,11], 
          [12,13], [13,12], [12,14], [14,12], [12,15], [15,12], [12,16], [16,12], [12,17], [17,12], [12,18], [18,12], [12,19], [19,12], [12,20], [20,12], 
          [13,14], [14,13], [13,15], [15,13], [13,16], [16,13], [13,18], [18,13],
          [14,15], [15,14], [14,16], [16,14], [14,26], [26,14],  
          [15,16], [16,15], [15,19], [19,15], 
          [16,20], [20,16], [16,27], [27,16], 
          [17,18], [18,17], [17,19], [19,17], [17,20], [20,17], [17,26], [26,17], 
          [18,19], [19,18], [18,20], [20,18], [18,26], [26,18],
          [19,20], [20,19], [19,26], [26,19], 
          [20,26], [26,20], 
          [21,22], [22,21], [21,26], [26,21], 
          [22,23], [23,22], [22,26], [26,22], [22,28], [28,22],
          [23,26], [26,23], [23,27], [27,23], [23,28], [28,23], 
          [24,26], [26,24]
            ], dtype=torch.long).t().to(self.device)
        
        self.edge_type = torch.tensor([
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,             # M0
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,   # M1
            0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1,                   # M2
            0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # M3
            0, 0, 1, 1, 0, 0, 0, 0, 1, 1,                         # M4
            0, 0, 0, 0, 1, 1,                                     # M5
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,                   # M6
            1, 1, 1, 1, 0, 0, 0, 0, 0, 0,                         # M7
            0, 0, 0, 0, 0, 0,                                     # M8
            0, 0, 0, 0,                                           # M9
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,                   # M10
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0,                         # M11
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,       # M12
            0, 0, 0, 0, 0, 0, 0, 0,                               # M13
            0, 0, 0, 0, 1, 1,                                     # M14
            0, 0, 0, 0,                                           # M15
            0, 0, 0, 0,                                           # M16
            0, 0, 0, 0, 0, 0, 1, 1,                               # M17
            0, 0, 0, 0, 1, 1,                                     # M18
            0, 0, 1, 1,                                           # M19
            1, 1,                                                 # M20
            0, 0, 1, 1,                                           # M21
            0, 0, 1, 1, 0, 0,                                     # M22
            1, 1, 0, 0, 0, 0,                                     # M23
            1, 1,                                                 # Ib
            ]).to(self.device)
        
        self.num_relations = 2
        self.num_nodes = 29
        self.num_node_features = 12
        self.obs_shape = (self.num_nodes, self.num_node_features)

        """Select an action from the input state."""

        self.L_C1 = 30 
        self.W_C1 = 30
        M_C1_low = 1
        M_C1_high = 40 
        self.C1_low = M_C1_low * (self.L_C1 * self.W_C1 * 2e-15 + (self.L_C1 + self.W_C1)*0.38e-15)
        self.C1_high = M_C1_high * (self.L_C1 * self.W_C1 * 2e-15 + (self.L_C1 + self.W_C1)*0.38e-15)
        
        self.W_C0 = 30
        self.L_C0 = 30
        M_C0_low = 1
        M_C0_high = 40
        self.C0_low = M_C0_low * (self.L_C0 * self.W_C0 * 2e-15 + (self.L_C0 + self.W_C0) *0.38e-15)
        self.C0_high = M_C0_high * (self.L_C0 * self.W_C0 * 2e-15 + (self.L_C0 + self.W_C0)*0.38e-15)
        
        self.action_space_low = np.array([ 0.5, 0.5, 1, # M0(W_low,L_low,M_low)
                                        0.5, 0.5, 1,    # M8
                                        0.5, 0.5, 1,    # M10
                                        0.5, 0.5, 1,   # M11
                                        0.5, 0.5, 1,    # M17
                                        0.5, 0.5, 1,    # M21
                                        0.5, 0.5, 1,    # M23
                                        1e-6,           # Ib
                                        M_C0_low,       # C0
                                        M_C1_low])      # C1
        
        self.action_space_high = np.array([10, 5, 50,  # M0(W_high,L_high,M_high) 
                                        10, 5, 50,     # M8  
                                        10, 5, 50,     # M10
                                        10, 5, 500,    # M11
                                        10, 5, 50,     # M17
                                        10, 5, 50,    # M21
                                        10, 5, 50,    # M23
                                        50e-6,        # Ib  
                                        M_C0_high,    # C0
                                        M_C1_high])   # C1
        
        self.action_dim = len(self.action_space_low)
        self.action_shape = (self.action_dim,)    
        
        """Some target specifications for the final design"""
        self.phase_margin_target = 60 
        self.CL = 100  #100pF
        self.dcgain_target = 130
        self.PSRP_target = -80
        self.PSRN_target = -80 
        self.cmrrdc_target = -80
        self.vos_target = 0.06e-3   
        self.TC_target = 10e-6
        self.settlingTime_target = 1e-6
        self.FOML_target = 160
        self.FOMS_target = 300
        self.Active_Area_target = 150
        self.Power_target = 0.3
        self.GBW_target = 1.2e6
        self.sr_target = 0.6        

        """ baseline """
        self.PSRR_base = -60
        self.cmrrdc_base = -60
        self.dcgain_base = 90
        self.FOMS_base = 200
        self.FOML_base = 60
        self.settlingTime_base = 5e-6
        self.Active_Area_base = 200
        self.TC_base = 50e-6
        self.vos_base = 0.1e-3
        self.Power_base = 0.5
        self.sr_base = 0.3
        self.GBW_base = 1e6

        self.GND = 0
        self.Vdd = 1.8       
        
class GraphLDOtestbench:
    """                                                                                                                           

    node 0 : M0 , node 1 : M1 , node 2 : M2 , node 3 : M3 , node 4 : M4 , node 5 : M5
    node 6 : M6 , node 7 : M7 , node 8 : M8 , node 9 : M9 , node 10 : M10 , node 11 : M11
    node 12 : M12 , node 13 : M13 , node 14 : M14 , node 15 : M15 , node 16 : M16 , node17 : M17 ,
    node18 : M18 , node19 : M19 , node20 : M20 , node21 : M21 , node22 : M22 , node23 : C0 , node24 : M24 , node25 : VDD ,
    node26 : GND , node27 : Ib , (node31 : vinp) , (node32 : vinn)  , (node33 : VDDP) , node28 : CL , node29 : R1 , node30 : R0

    """
    def __init__(self):        
        # self.device = torch.device(
        #     "cuda:0" if torch.cuda.is_available() else "cpu"
        # )
        
        self.device = torch.device(
           "cpu"
        )
        self.ckt_hierarchy = (
                      ('M0','x1.XM0','pfet_01v8','m'),
                      ('M1','x1.XM1','pfet_01v8','m'),
                      ('M2','x1.XM2','pfet_01v8','m'),
                      ('M3','x1.XM3','pfet_01v8','m'),
                      ('M4','x1.XM4','pfet_01v8','m'),
                      ('M5','x1.XM5','pfet_01v8','m'),
                      ('M6','x1.XM6','pfet_01v8','m'),
                      ('M7','x1.XM7','pfet_01v8','m'),
                      ('M8','x1.XM8','pfet_01v8','m'),
                      ('M9','x1.XM9','pfet_01v8','m'),
                      ('M10','x1.XM10','pfet_01v8_lvt','m'),
                      ('M24','x1.XM24','pfet_01v8','m'),
                      ('M11','x1.XM11','pfet_01v8','m'),
                      ('M12','x1.XM12','nfet_01v8','m'),
                      ('M13','x1.XM13','nfet_01v8','m'),
                      ('M14','x1.XM14','nfet_01v8','m'),
                      ('M15','x1.XM15','nfet_01v8','m'),
                      ('M16','x1.XM16','nfet_01v8','m'),
                      ('M17','x1.XM17','nfet_01v8','m'),
                      ('M18','x1.XM18','nfet_01v8','m'),
                      ('M19','x1.XM19','nfet_01v8','m'),
                      ('M20','x1.XM20','nfet_01v8','m'),
                      ('M21','x1.XM21','nfet_01v8','m'),
                      ('M22','x1.XM22','nfet_01v8','m'),

                      ('Ib','','Ib','i'),
                      ('C0','x1.XC0','cap_mim_m3_1','c'),
                      ('CL','XCL','cap_mim_m3_1','c')
                     )    

        self.op = {'M0':{},'M1':{},'M2':{},'M3':{},'M4':{},'M5':{},'M6':{},'M7':{},'M8':{}, 'M9':{},'M10':{},'M10':{},'M11':{},
                'M12':{},'M13':{},'M14':{},'M15':{},'M16':{},'M17':{},'M18':{},'M19':{},'M20':{},'M21':{},'M22':{},'M24':{},
                'Ib':{},'C0':{},'CL':{}
                 }

        self.edge_index = torch.tensor([
          [0,1], [1,0], [0,2], [2,0], [0,3], [3,0], [0,4], [4,0], [0,7], [7,0], [0,24], [24,0], [0,25], [25,0], [0,27], [27,0],
          [1,2], [2,1], [1,3], [3,1], [1,4], [4,1], [1,7], [7,1], [1,24], [24,1], [1,25], [25,1], [1,12], [12,1], 
          [1,27], [27,1], [1,17], [17,1], [1,18], [18,1], [1,19], [19,1], [1,20], [20,1], 
          [2,3], [3,2], [2,4], [4,2], [2,7], [7,2], [2,24], [24,2], [2,13], [13,2], [2,25], [25,2], [2,27], [27,2], 
          [3,4], [4,3], [3,7], [7,3], [3,24], [24,3], [3,25], [25,3], [3,27], [27,3], [3,12], [12,3], [3,13], [13,3], [3,14], [14,3], [3,15], [15,3], [3,16], [16,3], 
          [4,7], [7,4], [4,24], [24,4], [4,8], [8,4], [4,9], [9,4], [4,25], [25,4], [4,27], [27,4],
          [5,6], [6,5], [5,15], [15,5], [5,25], [25,5], 
          [6,15], [15,6], [6,16], [16,6], [6,10], [10,6], [6,25], [25,6], 
          [7,24], [24,7], [7,25], [25,7], [7,27], [27,7], [7,21], [21,7], [7,22], [22,7], 
          [24,25], [25,24], [24,10], [10,24], [24,27], [27,24], [24,11], [11,24], 
          [8,9], [9,8], [8,15], [15,8], [8,19], [19,8], 
          [9,16], [16,9], [9,20], [20,9], [9,23], [23,9], 
          [10,11], [11,10], [10,16], [16,10], [10,21], [21,10], 
          [11,25], [25,11], [11,29], [29,11], [11,23], [23,11], [11,28], [28,11], 
          [12,13], [13,12], [12,14], [14,12], [12,17], [17,12], [12,18], [18,12], [12,19], [19,12], [12,20], [20,12], [12,15], [15,12], [12,16], [16,12], 
          [13,14], [14,13], [13,18], [18,13], [13,15], [15,13], [13,16], [16,13], 
          [14,26], [26,14], [14,15], [15,14], [14,16], [16,14], 
          [15,16], [16,15], [15,19], [19,15], 
          [16,20], [20,16], [16,23], [23,16], 
          [17,18], [18,17], [17,19], [19,17], [17,20], [20,17], [17,26], [26,17], 
          [18,19], [19,18], [18,20], [20,18], [18,26], [26,18],
          [19,20], [20,19], [19,26], [26,19], 
          [20,23], [23,20], [20,26], [26,20], 
          [21,22], [22,21], [21,26], [26,21], 
          [22,26], [26,22],
          [23,28], [28,23], [23,29], [29,23], 
          [26,27], [27,26], [26,28], [28,26], [26,30], [30,26], 
          [28,29], [29,28],
          [29,30], [30,29], 

            ], dtype=torch.long).t().to(self.device)
        
        self.edge_type = torch.tensor([
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,       # M0
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,             # M1
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,             # M2
            0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # M3
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,                   # M4
            0, 0, 0, 0, 1, 1,                                     # M5
            0, 0, 0, 0, 0, 0, 1, 1,                               # M6
            0, 0, 1, 1, 1, 1, 0, 0, 0, 0,                         # M7
            1, 1, 0, 0, 1, 1, 0, 0,                               # M24
            0, 0, 0, 0, 0, 0,                                     # M8
            0, 0, 0, 0, 0, 0,                                     # M9
            0, 0, 0, 0, 0, 0,                                     # M10
            1, 1, 0, 0, 0, 0, 0, 0,                               # M11
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,       # M12
            0, 0, 0, 0, 0, 0, 0, 0,                               # M13
            1, 1, 0, 0, 0, 0,                                     # M14
            0, 0, 0, 0,                                           # M15
            0, 0, 0, 0,                                           # M16
            0, 0, 0, 0, 0, 0, 1, 1,                               # M17
            0, 0, 0, 0, 1, 1,                                     # M18
            0, 0, 1, 1,                                           # M19
            0, 0, 1, 1,                                           # M20
            0, 0, 1, 1,                                           # M21
            1, 1,                                                 # M22
            0, 0, 0, 0,                                           # C0
            1, 1, 1, 1, 1, 1,                                     # GND
            0, 0,                                                 # CL
            0, 0,                                                 # R1
            ]).to(self.device)
        
        self.num_relations = 2
        self.num_nodes = 31
        self.num_node_features = 14
        self.obs_shape = (self.num_nodes, self.num_node_features)

        """Select an action from the input state."""

        self.L_CL = 30 
        self.W_CL = 30
        M_CL_low = 1
        M_CL_high = 300 
        self.CL_low = M_CL_low * (self.L_CL * self.W_CL * 2e-15 + (self.L_CL + self.W_CL)*0.38e-15)
        self.CL_high = M_CL_high * (self.L_CL * self.W_CL * 2e-15 + (self.L_CL + self.W_CL)*0.38e-15)
    
        self.W_C0 = 10
        self.L_C0 = 10
        M_C0_low = 1
        M_C0_high = 50
        self.C0_low = M_C0_low * (self.L_C0 * self.W_C0 * 2e-15 + (self.L_C0 + self.W_C0) *0.38e-15)
        self.C0_high = M_C0_high * (self.L_C0 * self.W_C0*2e-15 + (self.L_C0 + self.W_C0)*0.38e-15)
        
        self.action_space_low = np.array([ 1, 0.5, 1,#M0(W_low,L_low,M_low)
                                        1, 0.5, 1, #M8
                                        1, 0.5, 1, #M10
                                        2, 0.15, 100, #M_power
                                        1, 0.5, 1, #M17
                                        1, 0.5, 1,#M21
                                        3e-6,  # Ib
                                        #M_Rfb_low, #Rfb
                                        M_C0_low, #C0
                                        M_CL_low]) # CL
        
        self.action_space_high = np.array([10, 5, 30,#M0(W_high,L_high,M_high) 
                                        10, 5, 30,#M8  
                                        10, 5, 30,#M10
                                        10, 2, 1000, #M_power
                                        10, 5, 30,#M17
                                        10, 5, 30,#M21  
                                        20e-6,   # Ib
                                        #M_Rfb_high,  
                                        M_C0_high,   # C0
                                        M_CL_high])  # CL
        
        self.action_dim = len(self.action_space_low)
        self.action_shape = (self.action_dim,)    
        
        self.Vout = 1.6
        self.Vref = 0.4
        self.GND = 0
        self.Vdd = 1.8
        self.r1=300e3
        self.r0=100e3

        """Some target specifications for the final design"""

        self.LDR_target = 0.1
        self.LNR_target = 0.01
        self.Power_maxload_target =  9e-5
        self.Power_minload_target =  9e-6
        self.vos_target = 2e-3

        self.PSRR_target = -40
        self.GBW_target = 2e6
        self.phase_margin_target = 60 

        self.v_undershoot_target = 0.1
        self.v_overshoot_target = 0.1

        self.rew_eng = True                