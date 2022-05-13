#!/bin/bash

#两集群位置
SHAOLIN="afs://shaolin.afs.baidu.com:9902"
TIANQI="afs://tianqi.afs.baidu.com:9902"

# 常用数据集位置

# --------- HDFS集群 -------------
# 公共根目录
KG="${SHAOLIN}/app/ecom/fengkong/kg/"

# 用户根目录
ZH="${SHAOLIN}/app/ecom/fengkong/personal/zhanghao55/"

# 工具包位置
TOOLS="${ZH}tools/"

# 任务根目录
PROJECTS="${ZH}projects/"

# 默认代码压缩包位置
JOB_ARCHIVES_ROOT="${PROJECTS}/default_job_archives/"

#切词和python包
WORDSEG_PATH="${TOOLS}wordseg.tar.gz"
WORD_SEG_DICT="${TOOLS}chinese_gbk.tar.gz"
HADOOP_PYTHON="${TOOLS}python_zhanghao55.tar.gz"
PROTO="${KG}tools/proto_stable.20211015.tar.gz"
PROTOBUF="${KG}tools/protobuf-3.17.3-py2.7.tar.gz"

# 本机python地址
LOCAL_PYTHON="/home/work/zhanghao55/tools/python_zhanghao55/bin/python"
#本机hadoop位置
HADOOP_HOME="/home/work/local/hadoop-client-1.6.2.2/hadoop/"
HADOOP_HOME="/home/work/cloud/hadoop-client/hadoop/"
HADOOP_CLIENT="${HADOOP_HOME}bin/hadoop --config ${HADOOP_HOME}conf-fengkong-afs-shaolin"

# 本机spark地址
SPARK_SUMBIT="/home/work/local/spark-2.3.2.9/bin/spark-submit"
SPARK_SUMBIT="/home/work/cloud/project/baidu/fengkong/def-monitor/analyse_feed/spark-2.3.2.9/bin/spark-submit"

# ======本地工作目录 BASE_DIR=$(dirname $(readlink -f "$0"))=======
BASE_DIR=$(cd "$(dirname "$0")";pwd)
DATA_DIR="${BASE_DIR}/../data/"
LOCAL_DATA_DIR="${BASE_DIR}/../local_data/"
LOG_DIR="${BASE_DIR}/../log/"
MODEL_DIR="${BASE_DIR}/../model/"
SRC_DIR="${BASE_DIR}/../src/"
OUTPUT_DIR="${BASE_DIR}/../output/"
BIN_DIR="${BASE_DIR}/../bin/"
LOG_FILE="util.log"

function make_dir()
{
    for cur_dir in $@
    do
        if [ ! -d ${cur_dir} ]; then
            echo "creat dir : ${cur_dir}"
            mkdir -p ${cur_dir}
        fi
    done
}

make_dir ${LOG_DIR}
make_dir ${OUTPUT_DIR}

function make_hdfs_dir()
{
    for hdfs_dir in $@
    do
        ${HADOOP_CLIENT} fs -test -e ${hdfs_dir}
        if [ $? -eq 1 ]; then
            ${HADOOP_CLIENT} fs -mkdir ${hdfs_dir}
        fi
    done
}

function log_info()
{
    info=${1}
    time_now=`date +"%Y%m%d %H:%M:%S"`
    echo -e "INFO, ${time_now}, $info}!" >> ${LOG_DIR}${LOG_FILE}
}

function exit_error()
{
    info=${1}
    time_now=`date +"%Y%m%d %H:%M:%S"`
    echo -e "ERROR, ${time_now}, ${info}!" >> ${LOG_DIR}${LOG_FILE}
    exit 1
}

function WriteLog()
{
    local msg_date=`date +%Y-%m-%d" "%H:%M:%S`
    local msg_begin=""
    local msg_end=""
    if [ $# -eq 1 ]; then
        local msg=$1
        echo "[${msg_date}]${msg}" | tee -a ${LOG_DIR}${LOG_FILE}
    elif [ $# -eq 2 ]
    then
        local msg=$2
        local runstat=$1
        if [ ${runstat} -eq 0 ]; then
            msg_begin="Success"
            msg_end="ok!"
        else
            msg_begin="Error"
            msg_end="fail!"
        fi
        echo "[${msg_date}][${msg_begin}]${msg} ${msg_end}" | tee -a ${LOG_DIR}${LOG_FILE}
        if [ ${runstat} -ne 0 ]; then
            echo "error when Task ${msg} runs at ${msg_date}" | tee -a ${LOG_DIR}${LOG_FILE}
            exit 1
        fi
    else
        return
    fi
}

# 分布式的def-user的文件可能存在多个集群的多个位置 因此对于某指定文件 要先找到其位置
function search_hdfs_file_path()
{
    # 数据类型：all.存量  click.点击
    local TYPE=${1}
    # 当前def-user的模型分为几块运行 当前为3
    local MAX_MODELS_SPLIT=${2}
    local FILE_NAME=${3}
    local P_FENGKONG="/app/ecom/aries/fengkong/def_model_data/def_user_mult/"
    local P_GALAXY="/app/ecom/aries/galaxy/def_model_data/def_user_mult/"
    local FILE_PATH=""
    local machines="${KHAN} ${MULAN}"
    # 首先加上 统一处理后的数据目录
    local paths="${P_FENGKONG}Z_${TYPE}/ ${P_GALAXY}Z_${TYPE}/"
    # 加上流程中间数据的输出目录
    # ${#paths[*]}: 当前数组paths的长度
    # paths[${#paths[*]}]=xxx即在该数组后面添加数据
    for ((i=1;i<=${MAX_MODELS_SPLIT};i++))
    do
        paths[${#paths[*]}]="${P_FENGKONG}P_${TYPE}_${i}/"
        paths[${#paths[*]}]="${P_GALAXY}P_${TYPE}_${i}/"
    done

    # 在各集群各目录寻找指定名称文件夹
    for m in ${machines[@]}
    do
        for p in ${paths[@]}
        do
            i_path=${m}${p}${FILE_NAME}
            ${HADOOP_CLIENT} fs -test -e ${i_path}
            if [ $? -eq 0 ]; then
                if [ "${FILE_PATH}"x == ""x ]; then
                    FILE_PATH="${i_path}/*"
                else
                    FILE_PATH="${FILE_PATH},${i_path}/*"
                fi
            fi
        done
    done
    WriteLog "get file path: "${FILE_PATH}
    echo -e ${FILE_PATH}
}

function clear_remote_env
{
    if [ "$1"x == ""x ]; then
        echo "remote env path is empty" |tee -a ${LOG_DIR}${LOG_FILE}
        exit 1
    fi

    ${HADOOP_CLIENT} fs -rmr $1
    ${HADOOP_CLIENT} fs -mkdir $1
}

function pack_dir
{
    #将directory中的文件打包到tmp文件夹中
    local directory=$1
    local tmp_dir=$2

    echo "handling $directory to $tmp_dir"
    #切换到directory中，将除.tar.gz的所有文件打包，然后退出文件夹（这个tar.gz就是以前打包的directory 不会有其他的tar文件了 但是下一句话不是移了吗？）
    cd $directory; tar zcvfh $directory.tar.gz * --exclude "*.tar.gz" --exclude "*.un~" --exclude "*.pyc" --exclude "*/.git*"; cd -
    #将directory中的打包好的tar.gz文件移到tmp文件夹中
    mv $directory/$directory.tar.gz $tmp_dir/
}

function prepair_localfile
{

    ARCHIVES_DIR=$1
    if [ "${ARCHIVES_DIR}"x == ""x ]; then
        echo "ARCHIVES_DIR is empty" |tee -a ${LOG_DIR}${LOG_FILE}
        exit 1
    fi
    shift 1
    rm -r tmp
    mkdir tmp

    #对于所有传入的文件夹名  将其打包加到tmp文件夹中
    for directory in $@; do
        pack_dir $directory tmp
    done

    # 打包上传所需所有文件
    for tar_file in $(ls tmp/*.tar.gz); do
        ${HADOOP_CLIENT} fs -put $tar_file ${ARCHIVES_DIR}
    done
}

function spark_process(){
    #demo: spark_process "yarn" "test_job" "input_path" "output_path" "python file" "executor_num"

    local MASTER=$1
    local TASK_NAME=$2
    local INPUT_FILE=$3
    local PYTHON_FILE=$4
    local EXECUTOR_NUM=$5
    local OUTPUT_FILE=$6
    local OUTPUT_PART_NUM=$7

    shift 7
    local OTHER_CONFIG=""
    for cur_config in $@; do
        OTHER_CONFIG="${OTHER_CONFIG} ${cur_config}"
    done

    WriteLog "input file   = ${INPUT_FILE}"
    WriteLog "python file  = ${PYTHON_FILE}"
    WriteLog "executor num = ${EXECUTOR_NUM}"
    WriteLog "output file  = ${OUTPUT_FILE}"
    WriteLog "output parts = ${OUTPUT_PART_NUM}"
    WriteLog "other config = ${OTHER_CONFIG}"

    if [[ ! ${OUTPUT_FILE} =~ kg ]] ; then
        echo "check your output file :"${OUTPUT_FILE}
        exit 1
    fi

    ${HADOOP_CLIENT} fs -test -e ${OUTPUT_FILE}
    if [ $? -eq 0 ]; then
        ${HADOOP_CLIENT} fs -rmr ${OUTPUT_FILE}
    fi

    ${SPARK_SUMBIT} \
        --master ${MASTER} \
        --queue "spark-fengkong"  \
        ${OTHER_CONFIG} \
        ${PYTHON_FILE} \
        ${TASK_NAME} \
        ${INPUT_FILE} \
        ${OUTPUT_FILE} \
        ${EXECUTOR_NUM} \
        ${OUTPUT_PART_NUM}

    WriteLog $? "spark job: ${TASK_NAME}"
}


function hadoop_process(){
    #parameter 1.process name 2.inputfile 3.outputfile 4.mapper name 5.mapper number 6.reducer name 7.reducer number
    ${HADOOP_CLIENT} fs -test -e $3
    if [ $? -eq 0 ]; then
        ${HADOOP_CLIENT} fs -rmr $3
    fi

    if [ "$4"x ==  "cat"x ]; then
        local MAPPER=$4
    else
        local MAPPER="python/bin/python "$4
    fi
    if [ "$6"x == "cat"x ]; then
        local REDUCER=$6
    else
        local REDUCER="python/bin/python "$6
    fi

    local TASK_NAME=$1
    local INPUT_FILE=$2
    local OUTPUT_FILE=$3
    local MAPPER_NUM=$5
    local REDUCER_NUM=$7

    if [[ ! ${OUTPUT_FILE} =~ kg ]] ; then
        echo "check your output file :"${OUTPUT_FILE}
        exit 1
    fi

    clear_remote_env ${JOB_ARCHIVES_ROOT}${TASK_NAME}
    shift 7
    local CACHE_ARCHIVES=""
    for cache_archive in $@; do
        cur_cache_dir="${JOB_ARCHIVES_ROOT}${TASK_NAME}/"
        prepair_localfile ${cur_cache_dir} ${cache_archive}
        CACHE_ARCHIVES=${CACHE_ARCHIVES}" -cacheArchive ${cur_cache_dir}${cache_archive}.tar.gz#${cache_archive}"
    done

    WriteLog "input file  = ${INPUT_FILE}"
    WriteLog "output file = ${OUTPUT_FILE}"
    WriteLog "mapper      = ${MAPPER}"
    WriteLog "reducer     = ${REDUCER}"
    ${HADOOP_CLIENT} streaming \
        -D mapred.job.queue.name="fengkong-galaxy-online_normal" \
        -D mapred.job.name=${TASK_NAME} \
        -D mapred.job.map.capacity=3000 \
        -D mapred.map.tasks=${MAPPER_NUM} \
        -D stream.memory.limit=5000 \
        -D mapred.reduce.tasks=${REDUCER_NUM} \
        -D mapred.map.over.capacity.allowed=false \
        -D mapred.job.priority=VERY_HIGH \
        -D abaci.job.base.environment=default \
        -D mapred.textoutputformat.ignoreseparator=true \
        -partitioner org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner \
        -inputformat org.apache.hadoop.mapred.TextInputFormat \
        -input ${INPUT_FILE} \
        -output ${OUTPUT_FILE} \
        -mapper "${MAPPER}" \
        -reducer "${REDUCER}" \
        -cacheArchive ${HADOOP_PYTHON}#python \
        -cacheArchive ${WORD_SEG_DICT}#dict \
        -cacheArchive ${WORDSEG_PATH}#wordseg \
        ${CACHE_ARCHIVES}

    if [ $? -ne 0 ]
        then
        echo "$1 error"
        exit 1
    fi
}

function hadoop_process_hce(){
    #parameter 1.process name 2.inputfile 3.outputfile 4.mapredname 5.mapper number 6.reducer number
    ${HADOOP_CLIENT} fs -test -e $3
    if [ $? -eq 0 ]; then
        ${HADOOP_CLIENT} fs -rmr $3
    fi

    local TASK_NAME=$1
    local INPUT_FILE=$2
    local OUTPUT_FILE=$3
    local MAPRED=$4
    local MAPPER_NUM=$5
    local REDUCER_NUM=$6

    if [[ ! ${OUTPUT_FILE} =~ kg ]] ; then
        echo "check your output file :"${OUTPUT_FILE}
        exit 1
    fi

    clear_remote_env ${JOB_ARCHIVES_ROOT}${TASK_NAME}
    shift 6
    local CACHE_ARCHIVES=""
    for cache_archive in $@; do
        cur_cache_dir="${JOB_ARCHIVES_ROOT}${TASK_NAME}/"
        prepair_localfile ${cur_cache_dir} ${cache_archive}
        CACHE_ARCHIVES=${CACHE_ARCHIVES}" -cacheArchive ${cur_cache_dir}${cache_archive}.tar.gz#${cache_archive}"
    done

    WriteLog "input file  = ${INPUT_FILE}"
    WriteLog "output file = ${OUTPUT_FILE}"
    WriteLog "mapred      = ${MAPRED}"
    ${HADOOP_CLIENT} hce \
        -D mapred.job.queue.name="fengkong-galaxy-online_normal" \
        -D mapred.job.name=${TASK_NAME} \
        -D mapred.job.map.capacity=3000 \
        -D mapred.map.tasks=${MAPPER_NUM} \
        -D stream.memory.limit=5000 \
        -D mapred.reduce.tasks=${REDUCER_NUM} \
        -D mapred.map.over.capacity.allowed=false \
        -D mapred.job.priority=VERY_HIGH \
        -D abaci.job.base.environment=default \
        -D mapred.textoutputformat.ignoreseparator=true \
        -inputformat SequenceFileInputFormat \
        -outputformat TextOutputFormat \
        -input ${INPUT_FILE} \
        -output ${OUTPUT_FILE} \
        -mapper "pyhce ${MAPRED}" \
        -reducer "pyhce ${MAPRED}" \
        ${CACHE_ARCHIVES}

    if [ $? -ne 0 ]
        then
        echo "$1 error"
        exit 1
    fi
}

#function hadoop_process_two_input(){
#    #parameter 1.process name 2.inputfile1 3.inputfile2 4.outputfile 5.mapper name 6.mapper number 7.reducer name 8.reducer number
#    ${HADOOP_CLIENT} fs -test -e $4
#    if [ $? -eq 0 ]; then
#        ${HADOOP_CLIENT} fs -rmr $4
#    fi
#
#    if [ "$5"x ==  "cat"x ]; then
#        local MAPPER=$5
#    else
#        local MAPPER="python/bin/python "$5
#    fi
#    if [ "$7"x == "cat"x ]; then
#        local REDUCER=$7
#    else
#        local REDUCER="python/bin/python "$7
#    fi
#
#    local TASK_NAME=$1
#    local INPUT_FILE1=$2
#    local INPUT_FILE2=$3
#    local OUTPUT_FILE=$4
#    local MAPPER_NUM=$6
#    local REDUCER_NUM=$8
#
#    if [[ ! ${OUTPUT_FILE} =~ zhanghao55 ]] ; then
#        echo "check your output file :"${OUTPUT_FILE}
#        exit 1
#    fi
#
#    clear_remote_env
#    shift 8
#    local CACHE_ARCHIVES=""
#    for cache_archive in $@; do
#        prepair_localfile ${cache_archive}
#        CACHE_ARCHIVES=${CACHE_ARCHIVES}" -cacheArchive ${JOB_ARCHIVES_ROOT}/${cache_archive}.tar.gz#${cache_archive}"
#    done
#
#    WriteLog "input file1 = "${INPUT_FILE1}
#    WriteLog "input file2 = "${INPUT_FILE2}
#    WriteLog "mapper = "${MAPPER}
#    WriteLog "reducer = "${REDUCER}
#    ${HADOOP_CLIENT} streaming \
#        -D mapred.job.queue.name="fengkong" \
#        -D mapred.job.name=${TASK_NAME} \
#        -D mapred.job.map.capacity=1000 \
#        -D mapred.map.tasks=${MAPPER_NUM} \
#        -D stream.memory.limit=5000 \
#        -D mapred.reduce.tasks=${REDUCER_NUM} \
#        -D mapred.map.over.capacity.allowed=false \
#        -D mapred.job.priority=VERY_HIGH \
#        -D abaci.job.base.environment=default \
#        -input ${INPUT_FILE1} \
#        -input ${INPUT_FILE2} \
#        -output ${OUTPUT_FILE} \
#        -mapper "${MAPPER}" \
#        -reducer "${REDUCER}" \
#        -cacheArchive ${HADOOP_PYTHON}#python \
#        -cacheArchive ${WORDSEG_PATH}#wordseg \
#        ${CACHE_ARCHIVES}
#
#    if [ $? -ne 0 ]
#        then
#        echo "$1 error"
#        exit 1
#    fi
#}
#
#function hadoop_process_with_conf_data_src(){
#    #parameter 1.process name 2.inputfile 3.outputfile 4.mapper name 5.mapper number 6.reducer name 7.reducer number
#    clear_remote_env && prepair_localfile src conf data
#
#    ${HADOOP_CLIENT} fs -test -e $3
#    if [ $? -eq 0 ]; then
#        ${HADOOP_CLIENT} fs -rmr $3
#    fi
#
#    if [ "$4"x ==  "cat"x ]; then
#        local MAPPER=$4
#    else
#        local MAPPER="python/bin/python "$4
#    fi
#    if [ "$6"x == "cat"x ]; then
#        local REDUCER=$6
#    else
#        local REDUCER="python/bin/python "$6
#    fi
#
#    echo "mapper = "${MAPPER}
#    echo "reducer= "${REDUCER}
#    ${HADOOP_CLIENT} streaming \
#        -D mapred.job.queue.name="fengkong" \
#        -D mapred.job.name=$1 \
#        -D mapred.job.map.capacity=1000 \
#        -D mapred.map.tasks=$5 \
#        -D stream.memory.limit=3000 \
#        -D mapred.reduce.tasks=$7 \
#        -D mapred.map.over.capacity.allowed=false \
#        -D mapred.job.priority=VERY_HIGH \
#        -D abaci.job.base.environment=default \
#        -input $2 \
#        -output $3 \
#        -mapper "${MAPPER}" \
#        -reducer "${REDUCER}" \
#        -cacheArchive ${HADOOP_PYTHON}#python \
#        -cacheArchive ${JOB_ARCHIVES_ROOT}/src.tar.gz#src \
#        -cacheArchive ${JOB_ARCHIVES_ROOT}/data.tar.gz#data \
#        -cacheArchive ${JOB_ARCHIVES_ROOT}/conf.tar.gz#conf
#
#    if [ $? -ne 0 ]
#        then
#        echo "$1 error"
#        exit 1
#    fi
#}
#
#function hadoop_process_with_cachefile(){
#    #parameter 1.process name 2.inputfile 3.outputfile 4.mapper name 5.mapper number 6.reducer name 7.reducer number
#    ${HADOOP_CLIENT} fs -test -e $3
#    if [ $? -eq 0 ]; then
#        ${HADOOP_CLIENT} fs -rmr $3
#    fi
#    if [ "$4"x ==  "cat"x ]; then
#        local MAPPER=$4
#    else
#        local MAPPER="python/bin/python "$4
#    fi
#    if [ "$6"x == "cat"x ]; then
#        local REDUCER=$6
#    else
#        local REDUCER="python/bin/python "$6
#    fi
#    local TASK_NAME=$1
#    local INPUT_FILE=$2
#    local OUTPUT_FILE=$3
#    local MAPPER_NUM=$5
#    local REDUCER_NUM=$7
#    shift 7
#    local CACHE_FILE=""
#    for cachefile in $@; do
#        CACHE_FILE=${CACHE_FILE}" -cacheFile "$cachefile
#    done
#
#    ${HADOOP_CLIENT} streaming \
#        -D mapred.job.queue.name="fengkong" \
#        -D mapred.job.name=${TASK_NAME} \
#        -D mapred.job.map.capacity=1000 \
#        -D mapred.map.tasks=${MAPPER_NUM} \
#        -D stream.memory.limit=3000 \
#        -D mapred.reduce.tasks=${REDUCER_NUM} \
#        -D mapred.map.over.capacity.allowed=false \
#        -D mapred.job.priority=VERY_HIGH \
#        -D abaci.job.base.environment=default \
#        -input ${INPUT_FILE} \
#        -output ${OUTPUT_FILE} \
#        -mapper "${MAPPER}" \
#        -reducer "${REDUCER}" \
#        -cacheArchive ${HADOOP_PYTHON}#python \
#        -cacheArchive ${JOB_ARCHIVES_ROOT}/src.tar.gz#src \
#        -cacheArchive ${JOB_ARCHIVES_ROOT}/data.tar.gz#data \
#        -cacheArchive ${JOB_ARCHIVES_ROOT}/conf.tar.gz#conf \
#        ${CACHE_FILE}
#
#    if [ $? -ne 0 ]
#        then
#        echo "${TASK_NAME} error"
#        exit 1
#    fi
#}

get_max_day_of_month()
{
    # 接收形如YYYYMM的日期
    local Y=`expr substr $1 1 4`
    local M=`expr substr $1 5 2`
    #取当月底最后一天
    local aa=`cal $M $Y` #日历
    local days=`echo $aa | awk '{print $NF}'`
    echo $days
}

last_X_days()
{
    local BEGIN_DATE=$1
    local MAX_DAY_AGO=7
    if [ "$2"x != ""x ]; then
        MAX_DAY_AGO=$2
    fi
    local DAYS_AGO=1
    local RES="{"${BEGIN_DATE}

    while [ ${DAYS_AGO} -lt ${MAX_DAY_AGO} ]; do
        RES=${RES}","`date -d "-${DAYS_AGO} day ${BEGIN_DATE}" +%Y%m%d`
        ((DAYS_AGO=DAYS_AGO+1))
    done
    RES=${RES}"}"

    echo ${RES}
}


last_X_months()
{
    local BEGIN_MONTH=$1
    local MAX_MONTH_AGO=6
    if [ "$2"x != ""x ]; then
        MAX_MONTH_AGO=$2
    fi
    local MONTHS_AGO=1
    local RES="{"${BEGIN_MONTH}

    while [ ${MONTHS_AGO} -lt ${MAX_MONTH_AGO} ]; do
        RES=${RES}","`date -d "-${MONTHS_AGO} month ${BEGIN_MONTH}01" +%Y%m`
        ((MONTHS_AGO=MONTHS_AGO+1))
    done
    RES=${RES}"}"

    echo ${RES}
}


days_range()
{
    # 日期格式YYYYMMDD
    local start_date=`date -d "${1}" +%Y%m%d`
    local end_date=`date -d "${2}" +%Y%m%d`
    local end_date_timestamp=`date -d "${end_date}" +%s`

    local data_list="{${start_date}"
    local cur_date=${start_date}
    local cur_date_timestamp=`date -d "${cur_date}" +%s`
    while [ ${cur_date_timestamp} -lt ${end_date_timestamp} ]
    do
        local cur_date=`date -d "+1 day ${cur_date}" +%Y%m%d`
        local cur_date_timestamp=`date -d "${cur_date}" +%s`
        data_list="${data_list},${cur_date}"
    done
    data_list=${data_list}"}"
    echo ${data_list}
}

function get_file_withmd5()
{
    local ftp_config=""
    if [ $# -eq 6 ]; then
        local ftp_config="--ftp-user=${1} --ftp-password=${2}"
        # 参数左移两位
        shift 2
    fi
    local ftp_path=${1}
    local ftp_name=${2}
    local dst_path=${3}
    local dst_name=${4}

    WriteLog "get_file_withmd5 ${ftp_name} begin."
    touch ${dst_path}${dst_name}.md5

    # 比对校验, 非生成校验
    wget -q --limit-rate=30m ${ftp_config} ${ftp_path}${ftp_name}.md5 -O \
        ${dst_path}${dst_name}.md5.new
    WriteLog $? "get ${ftp_name}.md5"

    diff ${dst_path}${dst_name}.md5 ${dst_path}${dst_name}.md5.new \
        > /dev/null
    if [ $? -eq 0 ]; then
        WriteLog "${dst_path}${dst_name} doesn't change"
    else
        wget -q --limit-rate=30m ${ftp_config} ${ftp_path}${ftp_name} -O \
            ${dst_path}${dst_name}
        WriteLog $? "get ${ftp_name}"
    fi
    mv ${dst_path}${dst_name}.md5.new ${dst_path}${dst_name}.md5
    WriteLog "get_file_withmd5 ${ftp_name} end."
}

function get_file_withoutmd5()
{

    local ftp_config=""
    if [ $# -eq 5 ]; then
        local ftp_config="--ftp-user=${1} --ftp-password=${2}"
        # 参数左移两位
        shift 2
    fi
    local ftp_path=${1}
    local ftp_name=${2}
    local dst_path=${3}
    local dst_name=${4}

    WriteLog "get_file_withoutmd5 ${ftp_name} begin."

    wget --limit-rate=30m ${ftp_config} ${ftp_path}${ftp_name} -O \
        ${dst_path}${dst_name} || \
        exit_error "get ${ftp_name} fail"

    WriteLog "get_file_withoutmd5 ${ftp_name} end."
}

init()
{

    # 不能用BIN DATA等定义的位置 因为此时是执行的公用的util.sh 而不是bin/util.sh
    make_dir bin data src log output local_data

    # 获取当前执行的util所在的绝对位置
    ln -s `cd "$(dirname "$0")"; pwd`"/util.sh" "bin/util.sh"

    # 获取当前执行的util所在的绝对位置
    cp `cd "$(dirname "$0")"; pwd`"/run_demo.sh" "bin/run.sh"
}

if [ "init"x == $1x ]; then
    init
fi
