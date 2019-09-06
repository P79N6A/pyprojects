import json
import gzip
import requests

data = {
	"data": [{
		"extra_status": {
			"state": 0,
			"scene": "unknown"
		},
		"service": "battery",
		"log_type": "performance_monitor",
		"session_id": "B6771B40-0BB6-4DFD-A09E-840E0083D732",
		"extra_values": {
			"level": -1
		},
		"timestamp": 1531883822036,
		"inapp_time": 0.0026628971099853516
	}, {
		"extra_status": {
			"state": 0,
			"scene": "unknown"
		},
		"service": "battery",
		"log_type": "performance_monitor",
		"session_id": "B306EC02-E995-4D4E-B6C3-B67540E6ACD6",
		"extra_values": {
			"level": -1
		},
		"timestamp": 1531883731914,
		"inapp_time": 0.00263214111328125
	}, {
		"extra_status": {
			"scene": "HMDViewController",
			"is_background": 0
		},
		"service": "cpu",
		"log_type": "performance_monitor",
		"session_id": "E71B1B17-329A-4F87-BE8A-D42FD6FE5B80",
		"extra_values": {
			"sys_usage": 0.029999999999999999,
			"user_usage": 0.042500000000000003,
			"app_usage": 0.0070000002160668373,
			"total_usage": 0.072500000000000009,
			"idle": 0.92749999999999999,
			"nice": 0
		},
		"timestamp": 1531883790914,
		"inapp_time": 43.898785829544067
	}, {
		"extra_status": {
			"scene": "HMDViewController",
			"is_background": 0
		},
		"service": "cpu",
		"log_type": "performance_monitor",
		"session_id": "B306EC02-E995-4D4E-B6C3-B67540E6ACD6",
		"extra_values": {
			"sys_usage": 0.034912718204488775,
			"user_usage": 0.034912718204488775,
			"app_usage": 0.0080000003799796104,
			"total_usage": 0.069825436408977551,
			"idle": 0.93017456359102246,
			"nice": 0
		},
		"timestamp": 1531883746914,
		"inapp_time": 15.003069877624512
	}, {
		"extra_status": {
			"scene": "HMDViewController"
		},
		"service": "fps",
		"log_type": "performance_monitor",
		"session_id": "E71B1B17-329A-4F87-BE8A-D42FD6FE5B80",
		"extra_values": {
			"fps": 59.999998799467015
		},
		"timestamp": 1531883791915,
		"inapp_time": 44.899505853652954
	}, {
		"extra_status": {
			"scene": "HMDViewController"
		},
		"service": "fps",
		"log_type": "performance_monitor",
		"session_id": "B306EC02-E995-4D4E-B6C3-B67540E6ACD6",
		"extra_values": {
			"fps": 59.99999880034013
		},
		"timestamp": 1531883746915,
		"inapp_time": 15.003343105316162
	}, {
		"session_id": "B306EC02-E995-4D4E-B6C3-B67540E6ACD6",
		"extra_values": {
			"didFinshedLaunching_to_first_render_time": 407.02998638153076,
			"from_load_to_didFinshedLaunching_time": 198.60196113586426,
			"from_load_to_first_render_time": 605.63194751739502
		},
		"timestamp": 1531883732322,
		"service": "start",
		"log_type": "performance_monitor"
	}, {
		"session_id": "B6771B40-0BB6-4DFD-A09E-840E0083D732",
		"extra_values": {
			"didFinshedLaunching_to_first_render_time": 641.88504219055176,
			"from_load_to_didFinshedLaunching_time": 289.84594345092773,
			"from_load_to_first_render_time": 931.73098564147949
		},
		"timestamp": 1531883822691,
		"service": "start",
		"log_type": "performance_monitor"
	}, {
		"extra_status": {
			"scene": "HMDViewController",
			"memory_warning": 0
		},
		"service": "memory",
		"log_type": "performance_monitor",
		"session_id": "E71B1B17-329A-4F87-BE8A-D42FD6FE5B80",
		"extra_values": {
			"total_memory": 17179869184,
			"availale_memory": 5686104064,
			"used_memory": 14000128000,
			"app_memory": 52019200
		},
		"timestamp": 1531883814931,
		"inapp_time": 67.916150808334351
	}, {
		"extra_status": {
			"scene": "HMDViewController",
			"memory_warning": 0
		},
		"service": "memory",
		"log_type": "performance_monitor",
		"session_id": "C139EF7B-3D6A-4DA9-B5D5-8910155C3518",
		"extra_values": {
			"total_memory": 17179869184,
			"availale_memory": 5517287424,
			"used_memory": 13734555648,
			"app_memory": 51421184
		},
		"timestamp": 1531883694770,
		"inapp_time": 15.022470951080322
	}, {
		"extra_status": {
			"scene": "HMDViewController",
			"memory_warning": 0
		},
		"service": "memory",
		"log_type": "performance_monitor",
		"session_id": "73F828E7-CF8F-4E4A-B442-F8054BB0072A",
		"extra_values": {
			"total_memory": 17179869184,
			"availale_memory": 5429485568,
			"used_memory": 13790445568,
			"app_memory": 52072448
		},
		"timestamp": 1531883722771,
		"inapp_time": 27.913141012191772
	}, {
		"extra_status": {
			"scene": "HMDViewController",
			"memory_warning": 0
		},
		"service": "memory",
		"log_type": "performance_monitor",
		"session_id": "B6771B40-0BB6-4DFD-A09E-840E0083D732",
		"extra_values": {
			"total_memory": 17179869184,
			"availale_memory": 5400596480,
			"used_memory": 14051946496,
			"app_memory": 51281920
		},
		"timestamp": 1531883835060,
		"inapp_time": 13.026394128799438
	}, {
		"extra_status": {
			"scene": "HMDViewController",
			"memory_warning": 0
		},
		"service": "memory",
		"log_type": "performance_monitor",
		"session_id": "B306EC02-E995-4D4E-B6C3-B67540E6ACD6",
		"extra_values": {
			"total_memory": 17179869184,
			"availale_memory": 6094786560,
			"used_memory": 14021570560,
			"app_memory": 51494912
		},
		"timestamp": 1531883746929,
		"inapp_time": 15.018025875091553
	}, {
		"host": "10.2.206.30",
		"session_id": "E71B1B17-329A-4F87-BE8A-D42FD6FE5B80",
		"timing_wait": 226,
		"receivedResponseContentLength": 1137,
		"responseHeader": "{\n  \"Cache-Control\" : \"must-revalidate,no-cache,no-store\",\n  \"Content-Type\" : \"text\\\/html;charset=ISO-8859-1\",\n  \"Content-Length\" : \"942\",\n  \"Server\" : \"Charles\",\n  \"Proxy-Connection\" : \"close\"\n}",
		"startTime": 1531883747439,
		"timing_receive": 4,
		"timing_isSocketReused": 0,
		"endtime": 1531883747696,
		"timing_ssl": 0,
		"upStreamBytes": 1075,
		"timing_connect": 2,
		"duration": 257,
		"network_type": 4,
		"inapp_time": 0.65123295783996582,
		"isSuccess": 0,
		"timing_proxy": 0,
		"errCode": 0,
		"protocolName": "http\/1.1",
		"timing_send": 1,
		"status": 503,
		"method": "POST",
		"MIMEType": "text\/html",
		"timing_isFromProxy": 0,
		"timing_remotePort": 0,
		"timing_dns": 0,
		"requestHeader": "{\n  \"Content-Encoding\" : \"gzip\",\n  \"Accept\" : \"application\\\/json\",\n  \"Content-Type\" : \"application\\\/json; encoding=utf-8\",\n  \"Content-Length\" : \"924\"\n}",
		"log_type": "api_error",
		"extra_status": {
			"scene": "HMDViewController"
		},
		"uri": "http:\/\/10.2.206.30:9317\/ios\/apm\/data\/add",
		"connetType": "WIFI",
		"timing_isCached": 0,
		"timestamp": 1531883747719
	}],
	"header": {
		"language": "en",
		"app_version": "1.0",
		"vendor_id": "94F35667-D943-4B7F-93F6-FAC3A63AD99E",
		"is_upgrade_user": False,
		"device_id": "123456789",
		"mcc_mnc": "",
		"resolution": "750*1334",
		"aid": "13",
		"os": "iOS",
		"update_version_code": "1.0.16",
		"access": "WIFI",
		"carrier": "",
		"crash_version": "1.0",
		"openudid": "951c41dc9ac9b83db2624b8302a6ab12ea151e4e",
		"timezone": 8,
		"is_jailbroken": False,
		"os_version": "11.4",
		"device_model": "x86_64 Simulator",
		"mc": "8C:85:90:A2:63:12",
		"display_name": "Heimdallr_Example",
		"package": "com.junyi.xie",
		"idfa": "4EC44BB3-C3A0-4AB6-9F04-FA1747DEFC42"
	}
}
# post_data = {'some': {'json': 'data'}}
header = {"content-encoding": "gzip"}
data = bytes(json.dumps(data), 'utf-8')
request_body = gzip.compress(data)
url = "http://10.8.78.136:9317/ios/apm/data/add"
r = requests.post(url, data=request_body, headers=header)