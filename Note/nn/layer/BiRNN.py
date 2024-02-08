import tensorflow as tf


class BiRNN:
    def __init__(self,fw_cells,bw_cells):
        # Receive a list of forward and backward RNNCell objects as parameters
        self.fw_cells=fw_cells
        self.bw_cells=bw_cells
        self.output_size=self.fw_cells.output_size+self.bw_cells.output_size
    
    
    def __call__(self,data):
        # Get batch_size from data
        batch_size=tf.shape(data)[0]
        # Reverse the input data along the time dimension to get the backward input data
        data_bw=tf.reverse(data,axis=[1])
        # Define a scan function to calculate the output and state of each time step
        def scan_fn(state,data,cell):
            output,state=cell.output(data,state)
            return output,state
        # Use tf.scan function to scan the forward and backward input data and get the forward and backward output data and state list
        outputs_fw=[]
        states_fw=[]
        outputs_bw=[]
        states_bw=[]
        for i in range(len(self.fw_cells)):
            cell=self.fw_cells[i]
            if i==0: # The first layer uses the original input data
                output_fw,state_fw=tf.scan(scan_fn,(data,cell),initializer=(tf.zeros([batch_size,32]),tf.zeros([batch_size,32])),swap_memory=True)
            else: # The later layers use the output data of the previous layer
                output_fw,state_fw=tf.scan(scan_fn,(outputs_fw[-1],cell),initializer=(tf.zeros([batch_size,32]),tf.zeros([batch_size,32])),swap_memory=True)
            outputs_fw.append(output_fw)
            states_fw.append(state_fw)
            cell=self.bw_cells[i]
            if i==0: # The first layer uses the reversed input data
                output_bw,state_bw=tf.scan(scan_fn,(data_bw,cell),initializer=(tf.zeros([batch_size,32]),tf.zeros([batch_size,32])),swap_memory=True)
            else: # The later layers use the output data of the previous layer
                output_bw,state_bw=tf.scan(scan_fn,(outputs_bw[-1],cell),initializer=(tf.zeros([batch_size,32]),tf.zeros([batch_size,32])),swap_memory=True)
            outputs_bw.append(output_bw)
            states_bw.append(state_bw)
        # Concatenate the forward and backward outputs and states to get the bidirectional outputs and states tensor
        output=tf.concat([outputs_fw[-1],outputs_bw[-1]],axis=-1) # Shape is [batch_size, seq_length, hidden_size * 2]
        state_fw=states_fw[-1] # Take the forward state of the last layer
        state_bw=states_bw[-1] # Take the backward state of the last layer
        state=tf.concat([state_fw[-1],state_bw[-1]],axis=-1) # Concatenate the forward and backward states of the last time step
        return output,state