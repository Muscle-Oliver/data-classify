# Data Type Classifier Deployment

**Demonstration of how to deploy HTTP services of data type classifier.**

Notes:

Once done step 1&2, a list of services can be seen in Clearml Web UI, under folder `DevOps`. These services need not to be modified afterwards.

Each time to deploy newly trained models, do step 3 only to update service endpoint.

## 1. Create a `clearml-serving` service
 
```shell
$ clearml-serving create --name "data classify"
```

Get the `id` of service "data classify": `8067b3e96aaa472aaaae944a69a7d790`

## 2. Start serving in k8s for data classify

Modify `clearml-serving/values.yaml`: 

Set `servingTaskId: 8067b3e96aaa472aaaae944a69a7d790`

(With superuser)
```shell
# kubectl create ns data-classify
# helm install -n data-classify clearml-serving-data-classify ./clearml-serving
```

## 3. Add model endpoint for the service

Manual deployment:

```shell
$ SERVICE_ID=8067b3e96aaa472aaaae944a69a7d790
$ clearml-serving --id $SERVICE_ID model add \
--engine triton \
--endpoint "data_classifier" \
--preprocess "inference_preprocess.py" \
--published \
--project "HPO Auto-training" \
--input-size 1 50 --input-name "INPUT__0" --input-type int64 \
--output-size 1 11 --output-name "OUTPUT__0" --output-type float32
```

**or**

Automatic deployment:

```shell
$ SERVICE_ID=8067b3e96aaa472aaaae944a69a7d790
$ clearml-serving --id $SERVICE_ID model auto-update \
--engine triton \
--endpoint "data_classifier_auto" \
--preprocess "inference_preprocess.py" \
--published \
--max-versions 2 \
--project "HPO Auto-training" \
--input-size 1 50 --input-name "INPUT__0" --input-type int64 \
--output-size 1 11 --output-name "OUTPUT__0" --output-type float32
```

## 4. Try HTTP request

```shell
$ curl -X POST \
  http://192.168.148.139:31370/serve/data_classifier \
  -H 'cache-control: no-cache' \
  -H 'content-type: application/json' \
  -d '{"text": "姚明"}'
```

## 5. (Optional) Add metrics to Grafana/Prometheus

```shell

```