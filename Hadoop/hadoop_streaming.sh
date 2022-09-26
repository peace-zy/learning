/ssd2/xx/3rdparty/hadoop-client/hadoop/bin/hadoop dfs -rmr afs://shaolin.afs.baidu.com:9902/app/ecom/fengkong/personal/xx/deepblue_pb
HADOOP_CLIENT="/ssd2/xx/3rdparty/hadoop-client/hadoop/bin/hadoop --config conf-fengkong-afs-shaolin"
#-input "afs://shaolin.afs.baidu.com:9902/app/ecom/fengkong/personal/hekaiwen/fc.vcg.2021.res" \
${HADOOP_CLIENT} streaming \
    -D mapred.job.queue.name="fengkong-galaxy-online_normal" \
    -D mapred.job.name="generate_pb" \
    -D mapred.job.map.capacity=3000 \
    -D mapred.map.tasks=1 \
    -D stream.memory.limit=5000 \
    -D mapred.reduce.tasks=10 \
    -D mapred.job.priority=VERY_HIGH \
    -D abaci.job.base.environment=default \
    -D mapred.textoutputformat.ignoreseparator=true \
    -partitioner org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner \
    -inputformat org.apache.hadoop.mapred.TextInputFormat \
    -input "afs://shaolin.afs.baidu.com:9902/app/ecom/fengkong/personal/liuxing07/emb_data/sample_pic_3M.url_localpath.crops.localurl.cspd_image_similarity.valid" \
    -output "afs://shaolin.afs.baidu.com:9902/app/ecom/fengkong/personal/xx/deepblue_pb" \
    -mapper "python/bin/python script/gen_docpb_for_hadoop.py" \
    -reducer "python/bin/python script/filter_first_col.py" \
    -cacheArchive "afs://shaolin.afs.baidu.com:9902/app/ecom/fengkong/kg/tools/python2.7.tar.gz"#python \
    -cacheArchive "afs://shaolin.afs.baidu.com:9902/app/ecom/fengkong/personal/xx/script.tar.gz"#script
