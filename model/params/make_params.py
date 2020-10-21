#! /usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Prints files with different combinations of parameters to be used to define different models.
'''

batch_size=[16,32,48]
filters = [10,20]
num_res_blocks=[1,2]
dilation_rate = [3,5]
step_size=[10,20,40]
num_cycles=[5,10]

for bs in batch_size:
    for fil in filters:
        for block in num_res_blocks:
            for dr in dilation_rate:
                for ss in step_size:
                    for n in num_cycles:
                        name = str(bs)+'_'+str(fil)+'_'+str(block)+'_'+str(dr)+'_'+str(ss)+'_'+str(n)+'.params'
                        with open(name, "w") as file:
                            file.write('batch_size='+str(bs)+'\n')
                            file.write('filters='+str(fil)+'\n')
                            file.write('num_res_blocks='+str(block)+'\n')
                            file.write('dilation_rate='+str(dr)+'\n')
                            file.write('step_size='+str(ss)+'\n')
                            file.write('num_cycles='+str(n)+'\n')
