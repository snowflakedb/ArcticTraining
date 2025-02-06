LOCAL_HOSTFILE='/data-fast/hostfile'

if [ -e ${HOSTFILE} ]; then
    HOSTFILE_TEMP='/code/users/samyam/hostfile_temp'
    cp ${LOCAL_HOSTFILE} ${HOSTFILE_TEMP}
    HOSTFILE=${HOSTFILE_TEMP}
fi

if [ -e ${HOSTFILE} ]; then
    ds_ssh -f /data-fast/hostfile bash `pwd`/vllm_teardown.sh
else
    bash `pwd`/vllm_teardown.sh
fi
