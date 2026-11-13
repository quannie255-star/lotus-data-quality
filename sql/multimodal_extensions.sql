-- sql/multimodal_extensions.sql — metastore 追加列（M11 多模态扩展，向后兼容）
--
-- 用法（在 repo 根目录）：
--   docker compose cp sql/multimodal_extensions.sql spark-master:/tmp/mm.sql
--   docker compose exec -T spark-master /opt/spark/bin/spark-sql \
--       --conf spark.sql.hive.manageFilesourcePartitions=false -f /tmp/mm.sql
--
-- 四个必须知道的约束（与 raw_docs.sql 同源的经验 + 多模态特有语义）：
--
-- 1. **只追加，不修改**：modality / modality_pair_id 均为新增列，历史 parquet 文件
--    没有这两列——读出 NULL。这与 spine「未检测 ≠ 空 ≠ 有值」三态语义一致：
--    NULL = 该行入库时此概念不存在，不是「检测了且为空」（M1-D4 的列级推广）。
--
-- 2. **modality 是列不是分区**：分区仍只按 source（WP-01 §1 分区策略不变，无宪法
--    豁免）。理由：模态数少且查询常与 source 正交，为省几个文件扫描引入第二分区维
--    会让「阶段只追加不重排」的朴素性破产。
--
-- 3. **不改表位置/不动已有列**：ALTER TABLE ADD COLUMNS 只扩 metastore 侧 schema，
--    已有数据的指纹与哈希不受影响（本 DDL 不触碰 data/ 产物；产物列扩展随
--    各阶段写入的新 part 文件自然携带）。
--
-- 4. **分区发现仍走文件系统**（manageFilesourcePartitions=false 语义同 raw_docs.sql），
--    新列对分区发现零影响。

ALTER TABLE raw_docs ADD COLUMNS IF NOT EXISTS (
  modality string COMMENT '样本模态: text/image/image_text/audio/video; 历史行=NULL(未检测)',
  modality_pair_id string COMMENT '跨模态配对组ID; 单模态行=NULL'
);

-- 验收：历史行新列必须读出 NULL（而非空串），计数不丢
-- SELECT modality, COUNT(*) FROM raw_docs GROUP BY modality;   -- 期望: NULL 231314
