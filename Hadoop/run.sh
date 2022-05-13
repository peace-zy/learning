#!/bin/bash

source ../parse_audit_log/bin/util.sh

# ----------------- 集群目录参数 -----------------------
INPUT_ROOT="${KG}parse_audit_log"
HDFS_ROOT="${KG}analyse_jimuyu"
MEG_ROOT="${SHAOLIN}/app/ecom/fengkong/dw/fengkong_data/userid_new_ssg_trade_day/"

TODAY=`date +%Y%m%d`
YESTERDAY=`date -d "-1 day" +%Y%m%d`

#确定log文件位置
LOG_FILE=${TODAY}_run.log

function load_feed_meg(){
    WriteLog "load_feed_meg start..."
    # 行业信息路径
    INPUT_MEG="${MEG_ROOT}time_stamp=${YESTERDAY}000000/000000_0"

    ${HADOOP_CLIENT} fs -cat ${INPUT_MEG} | iconv -c -f utf8 -t gbk > ${DATA_DIR}/feed_meg.txt

    WriteLog "load_feed_meg end."
}

#基木鱼日志
function parse_jimuyu_audit_log(){
    WriteLog "parse_jimuyu_audit_log start."

    if [[ "$1"x == ""x ]]; then
        echo "begin date of the parse_jimuyu_audit_log should be specified"
        exit 1
    fi
    if [[ "$2"x == ""x ]]; then
        echo "end date of the parse_jimuyu_audit_log should be specified"
        exit 1
    fi
    BEGIN_DAY=$1
    END_DAY=$2
    date_array=`days_range ${BEGIN_DAY} ${END_DAY}`

    # 输入路径
    INPUT=${INPUT_ROOT}"/parse_ad_audit_log_${date_array}"

    # 任务名称
    JOB_NAME="parse_jimuyu_audit_log_${BEGIN_DAY}_${END_DAY}"

    # 输出路径
    OUTPUT=${HDFS_ROOT}/${JOB_NAME}

    hadoop_process "${JOB_NAME}_wangchanghai01" \
        ${INPUT} \
        ${OUTPUT} \
        src/parse_jimuyu_audit_log.py 2000 \
        src/filter_first_col.py 1000 \
        src \
        data \
        > "${LOG_DIR}/${JOB_NAME}_hadoop.log" 2>&1
    WriteLog $? "${JOB_NAME}_hadoop"

    WriteLog "parse_jimuyu_audit_log end."

    ${HADOOP_CLIENT} fs -cat "${OUTPUT}/*" > "${OUTPUT_DIR}${JOB_NAME}.txt"
}

# 基木鱼日志统计（机器通过、拒绝，人工通过、拒绝）
function static_by_overall_ad(){
    WriteLog "static_by_overall_ad start."

    if [[ "$1"x == ""x ]]; then
        echo "begin date of the static_by_overall_ad should be specified"
        exit 1
    fi
    if [[ "$2"x == ""x ]]; then
        echo "end date of the static_by_overall_ad should be specified"
        exit 1
    fi
    BEGIN_DAY=$1
    END_DAY=$2
    date_array=`days_range ${BEGIN_DAY} ${END_DAY}`

    INPUT=${HDFS_ROOT}"/parse_jimuyu_audit_log_${BEGIN_DAY}_${END_DAY}/*"
    ${HADOOP_CLIENT} fs -cat ${INPUT} | \
        awk -F'\t' '{
                        f[$14"\t"$18]+=1;sum+=1
                    }END{
                        for(s in f)
                            print s"\t"f[s]"\t"f[s]/sum
                    }' | \
        awk -F'\t' 'ARGIND==1{
                        f[$1"\t"$2]=$3
                    }
                    ARGIND==2{
                        if($1"\t"$2 in f)
                            print f[$1"\t"$2]"\t"$3"\t"$4
                    }' ${DATA_DIR}"type_map.txt" - > \
    ${OUTPUT_DIR}"static_by_overall_ad_${BEGIN_DAY}_${END_DAY}.txt"

    ${HADOOP_CLIENT} fs -cat ${INPUT} | wc -l | \
        awk '{print "total:"$1}' >> ${OUTPUT_DIR}"static_by_overall_ad_${BEGIN_DAY}_${END_DAY}.txt"

    WriteLog "static_by_overall_ad start."
}

# 基木鱼机审拒绝
function parse_machine_reject(){
    WriteLog "parse_machine_reject start."

    if [[ "$1"x == ""x ]]; then
        echo "begin date of the parse_machine_reject should be specified"
        exit 1
    fi
    if [[ "$2"x == ""x ]]; then
        echo "end date of the parse_machine_reject should be specified"
        exit 1
    fi
    BEGIN_DAY=$1
    END_DAY=$2
    date_array=`days_range ${BEGIN_DAY} ${END_DAY}`

    # 输入路径
    INPUT=${HDFS_ROOT}"/parse_jimuyu_audit_log_${BEGIN_DAY}_${END_DAY}"

    # 任务名称
    JOB_NAME="parse_machine_reject_${BEGIN_DAY}_${END_DAY}"

    # 输出路径
    OUTPUT=${HDFS_ROOT}/${JOB_NAME}

    hadoop_process "${JOB_NAME}_wangchanghai01" \
        ${INPUT} \
        ${OUTPUT} \
        src/parse_machine_reject.py 1000 \
        cat 10 \
        src \
        data \
        > "${LOG_DIR}/${JOB_NAME}_hadoop.log" 2>&1
    WriteLog $? "${JOB_NAME}_hadoop"

    WriteLog "parse_machine_reject end."

    ${HADOOP_CLIENT} fs -cat "${OUTPUT}/*" > "${OUTPUT_DIR}${JOB_NAME}.txt"
}

# 基木鱼人审拒绝
function parse_human_reject(){
    WriteLog "parse_human_reject start."

    if [[ "$1"x == ""x ]]; then
        echo "begin date of the parse_human_reject should be specified"
        exit 1
    fi
    if [[ "$2"x == ""x ]]; then
        echo "end date of the parse_human_reject should be specified"
        exit 1
    fi
    BEGIN_DAY=$1
    END_DAY=$2
    date_array=`days_range ${BEGIN_DAY} ${END_DAY}`

    # 输入路径
    INPUT=${HDFS_ROOT}"/parse_jimuyu_audit_log_${BEGIN_DAY}_${END_DAY}"

    # 任务名称
    JOB_NAME="parse_human_reject_${BEGIN_DAY}_${END_DAY}"

    # 输出路径
    OUTPUT=${HDFS_ROOT}/${JOB_NAME}

    hadoop_process "${JOB_NAME}_wangchanghai01" \
        ${INPUT} \
        ${OUTPUT} \
        src/parse_human_reject.py 1000 \
        cat 10 \
        src \
        data \
        > "${LOG_DIR}/${JOB_NAME}_hadoop.log" 2>&1
    WriteLog $? "${JOB_NAME}_hadoop"

    WriteLog "parse_human_reject end."

    ${HADOOP_CLIENT} fs -cat "${OUTPUT}/*" > "${OUTPUT_DIR}${JOB_NAME}.txt"
}

# 基木鱼日志抽样(物料维度数据)
function sample_by_overall_ad(){
    WriteLog "sample_by_overall_ad start."

    if [[ "$1"x == ""x ]]; then
        echo "begin date of the sample_by_overall_ad should be specified"
        exit 1
    fi
    if [[ "$2"x == ""x ]]; then
        echo "end date of the sample_by_overall_ad should be specified"
        exit 1
    fi
    BEGIN_DAY=$1
    END_DAY=$2

    date_array=`days_range ${BEGIN_DAY} ${END_DAY}`

    # 输入路径
    INPUT=${HDFS_ROOT}"/parse_jimuyu_audit_log_${BEGIN_DAY}_${END_DAY}"

    # 任务名称
    JOB_NAME="sample_by_overall_ad_${BEGIN_DAY}_${END_DAY}"

    # 输出路径
    OUTPUT=${HDFS_ROOT}/${JOB_NAME}

    hadoop_process "${JOB_NAME}_wangchanghai01" \
        ${INPUT} \
        ${OUTPUT} \
        cat 5000 \
        src/sample_by_overall_ad.py 1 \
        src \
        data \
        > "${LOG_DIR}/${JOB_NAME}_hadoop.log" 2>&1
    WriteLog $? "${JOB_NAME}_hadoop"

    WriteLog "sample_by_overall_ad end."

    ${HADOOP_CLIENT} fs -cat "${OUTPUT}/*" > "${OUTPUT_DIR}${JOB_NAME}.txt"
}

# 基木鱼人审拒绝抽样
function sample_by_human_reject(){
    WriteLog "sample_by_human_reject start."

    if [[ "$1"x == ""x ]]; then
        echo "begin date of the sample_by_human_reject should be specified"
        exit 1
    fi
    if [[ "$2"x == ""x ]]; then
        echo "end date of the sample_by_human_reject should be specified"
        exit 1
    fi
    BEGIN_DAY=$1
    END_DAY=$2

    date_array=`days_range ${BEGIN_DAY} ${END_DAY}`

    # 输入路径
    INPUT=${HDFS_ROOT}"/parse_jimuyu_audit_log_${BEGIN_DAY}_${END_DAY}"

    # 任务名称
    JOB_NAME="sample_by_human_reject_${BEGIN_DAY}_${END_DAY}"

    # 输出路径
    OUTPUT=${HDFS_ROOT}/${JOB_NAME}

    hadoop_process "${JOB_NAME}_wangchanghai01" \
        ${INPUT} \
        ${OUTPUT} \
        cat 5000 \
        src/sample_by_human_reject.py 1 \
        src \
        data \
        > "${LOG_DIR}/${JOB_NAME}_hadoop.log" 2>&1
    WriteLog $? "${JOB_NAME}_hadoop"

    WriteLog "sample_by_human_reject end."

    ${HADOOP_CLIENT} fs -cat "${OUTPUT}/*" > "${OUTPUT_DIR}${JOB_NAME}.txt"
}

function main(){
    load_feed_meg
    # $1-$2基木鱼日志
    parse_jimuyu_audit_log $1 $2
    # $1-$2基木鱼日志统计（机器通过、拒绝，人工通过、拒绝）
    static_by_overall_ad $1 $2

    # $1-$2基木鱼机审拒绝
    parse_machine_reject $1 $2
    # $1-$2基木鱼人审拒绝
    parse_human_reject $1 $2

    # $1-$2基木鱼日志抽样
    sample_by_overall_ad $1 $2
    # $1-$2基木鱼人审拒绝抽样
    sample_by_human_reject $1 $2
}
