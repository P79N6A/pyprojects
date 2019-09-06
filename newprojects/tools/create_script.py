#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import sys

file_path = "/home/tiger/xiongsilan/gatling-new/xiongsilan-xgLive/galting/user-files/simulations/computerdatabase/XGLive/rooms"
temp_file_path = "/Users/withheart/Documents/studys/tools"


class WriteScripts:
    file_name = {}
    file_name['rooms'] = 'rooms.scala'
    file_name['audience_room_info'] = 'audience_room_info.scala'
    file_name['get_msg'] = 'get_msg.scala'
    file_name['rank_list'] = 'rank_list.scala'
    file_name['chat'] = 'chat.scala'
    room_id = '6627621923931032324'

    stage_params = {
        'rooms': 120,
        'audience_room_info': 1500,
        'get_msg': 20000,
        'rank_list': 8000,
        'chat': 15
    }

    script_front = {}
    script_end = ').protocols(http_conf))\n' \
                       '}\n'

    common_input = 'import java.util\n' \
                   'import java.util.Collections\n' \
                   'import java.util.concurrent.ThreadLocalRandom\n' \
                   'import io.gatling.core.Predef._\n' \
                   'import io.gatling.http.Predef._\n' \
                   'import scala.collection.mutable.ArraySeq\n' \
                   'import io.gatling.core.structure.PopulationBuilder\n' \
                   'import org.asynchttpclient.{Param, RequestBuilderBase}\n'

    script_front['rooms'] = 'package computerdatabase.RoomsTest\n' + common_input + \
                   'class RoomsTest extends Simulation {\n' \
                   'var http_conf = http.baseURL("http://i.snssdk.com")\n' \
                   'val didFeeder = csv("did.csv").random\n' \
                   'val scn = scenario("room/enter").feed(didFeeder).\n' \
                   '		exec(http("room/enter").\n' \
                   '		post("/videolive/room/enter?room_id='+room_id+'&iid=46732565125&device_id=${did}&ac=wifi&channel=huawei&aid=32&app_name=video_article&version_code=702&version_name=7.0.2&device_platform=android&ab_version=544933%2C321290%2C425684%2C549910%2C545153%2C554456%2C437002%2C539907%2C521142%2C544698%2C521140%2C357767%2C551966%2C523413%2C542501%2C548493%2C442144%2C487822%2C374103%2C536394%2C489304%2C539966%2C378354%2C553336%2C552063%2C459657%2C506536%2C545048%2C553899%2C457536%2C543415%2C552304%2C551242%2C344692&ssmix=a&device_type=FRD-AL10&device_brand=honor&language=zh&os_api=24&os_version=7.0&uuid=863127036074315&openudid=be61d330e8faf172&manifest_version_code=302&resolution=1080*1794&dpi=480&update_version_code=70208&_rticket=1539941348216&fp=LlTqL2X5crDr").\n' \
                   '		header("X-Tt-Stress","xigua_live_test").\n' \
                   '		check(status.is(200))).\n' \
                   '		exec(http("room/leave").\n' \
                   '		post("/videolive/room/leave?room_id='+room_id+'&iid=46732565125&device_id=${did}&ac=wifi&channel=huawei&aid=32&app_name=video_article&version_code=702&version_name=7.0.2&device_platform=android&ab_version=544933%2C321290%2C425684%2C549910%2C545153%2C554456%2C437002%2C539907%2C521142%2C544698%2C521140%2C357767%2C551966%2C523413%2C542501%2C548493%2C442144%2C487822%2C374103%2C536394%2C489304%2C539966%2C378354%2C553336%2C552063%2C459657%2C506536%2C545048%2C553899%2C457536%2C543415%2C552304%2C551242%2C344692&ssmix=a&device_type=FRD-AL10&device_brand=honor&language=zh&os_api=24&os_version=7.0&uuid=863127036074315&openudid=be61d330e8faf172&manifest_version_code=302&resolution=1080*1794&dpi=480&update_version_code=70208&_rticket=1539941638847&fp=LlTqL2X5crDr").\n' \
                   '		header("X-Tt-Stress","xigua_live_test").\n' \
                   '		check(status.is(200)))\n' \
                   '	setUp(scn.inject(\n'

    script_front['audience_room_info'] = 'package computerdatabase.AudienceRoomTest\n' + common_input + \
                                      'class AudienceRoomTest extends Simulation {\n' \
                                      'var http_conf = http.baseURL("http://i.snssdk.com")\n' \
                                      'val didFeeder = csv("did.csv").random\n' \
                                      'val scn = scenario("AudienceRoomTest").feed(didFeeder).\n' \
                                      '			exec(http("AudienceRoomTest").\n' \
                                      '			get("/videolive/user/audience_room_info?room_id='+room_id+'&&iid=46732565125&device_id=${did}&ac=wifi&channel=huawei&aid=32&app_name=video_article&version_code=702&version_name=7.0.2&device_platform=android&ab_version=544933%2C321290%2C425684%2C549910%2C545153%2C554456%2C437002%2C539907%2C521142%2C544698%2C521140%2C357767%2C551966%2C523413%2C542501%2C548493%2C442144%2C487822%2C374103%2C536394%2C489304%2C539966%2C378354%2C553336%2C552063%2C459657%2C506536%2C545048%2C553899%2C457536%2C543415%2C552304%2C551242%2C344692&ssmix=a&device_type=FRD-AL10&device_brand=honor&language=zh&os_api=24&os_version=7.0&uuid=863127036074315&openudid=be61d330e8faf172&manifest_version_code=302&resolution=1080*1794&dpi=480&update_version_code=70208&_rticket=1539941638847&fp=LlTqL2X5crDr").\n' \
                                      '			header("X-Tt-Stress","xigua_live_test").\n' \
                                      '			check(status.is(200)))\n' \
                                      '	setUp(scn.inject(\n'


    script_front['get_msg'] = 'package computerdatabase.GetMsgTest\n' + common_input + \
                           'class GetMsgTest extends Simulation {\n' \
                           'var http_conf = http.baseURL("http://i.snssdk.com")\n' \
                           'val didFeeder = csv("did.csv").random\n' \
                           'val scn = scenario("GetMsgTest").feed(didFeeder).\n' \
                           '				exec(http("GetMsgTest").\n' \
                           '				get("/videolive/im/get_msg?cursor=0&room_id='+room_id+'&&sdk_version=120&iid=46732565125&device_id=${did}&ac=wifi&channel=huawei&aid=32&app_name=video_article&version_code=702&version_name=7.0.2&device_platform=android&ab_version=544933%2C321290%2C425684%2C549910%2C545153%2C554456%2C437002%2C539907%2C521142%2C544698%2C521140%2C357767%2C551966%2C523413%2C542501%2C548493%2C442144%2C487822%2C374103%2C536394%2C489304%2C539966%2C378354%2C553336%2C552063%2C459657%2C506536%2C545048%2C553899%2C457536%2C543415%2C552304%2C551242%2C344692&ssmix=a&device_type=FRD-AL10&device_brand=honor&language=zh&os_api=24&os_version=7.0&uuid=863127036074315&openudid=be61d330e8faf172&manifest_version_code=302&resolution=1080*1794&dpi=480&update_version_code=70208&_rticket=1539941348352&fp=LlTqL2X5crDrFlHuJ2U1FlceL2cu&rom_version=EmotionUI_5.0_FRD-AL10C00B387").\n' \
                           '				header("X-Tt-Stress","xigua_live_test").\n' \
                           '				check(status.is(200)))\n' \
                           '	setUp(scn.inject(\n'


    script_front['rank_list'] = 'package computerdatabase.RanklistTest\n' + common_input + \
                             'class RanklistTest extends Simulation {\n' \
                             'val didFeeder = csv("did.csv").random\n' \
                             'var http_conf = http.baseURL("http://i.snssdk.com")\n' \
                             'val scn = scenario("RanklistTest").feed(didFeeder).\n' \
                             '		exec(http("RanklistTest").\n' \
                             '		get("/videolive/room/get_rank_list_v2?room_id='+room_id+'&rank_type=9&is_entrance=false&iid=46732565125&device_id=${did}&ac=wifi&channel=huawei&aid=32&app_name=video_article&version_code=702&version_name=7.0.2&device_platform=android&ab_version=544933%2C321290%2C425684%2C549910%2C545153%2C554456%2C437002%2C539907%2C521142%2C544698%2C521140%2C357767%2C551966%2C523413%2C542501%2C548493%2C442144%2C487822%2C374103%2C536394%2C489304%2C539966%2C378354%2C553336%2C552063%2C459657%2C506536%2C545048%2C553899%2C457536%2C543415%2C552304%2C551242%2C344692&ssmix=a&device_type=FRD-AL10&device_brand=honor&language=zh&os_api=24&os_version=7.0&uuid=863127036074315&openudid=be61d330e8faf172&manifest_version_code=302&resolution=1080*1794&dpi=480&update_version_code=70208&_rticket=1539941348514&fp=LlTqL2X5crDrFlHuJ2U1FlceL2cu&rom_version=EmotionUI_5.0_FRD-AL10C00B387").\n' \
                             '		header("X-Tt-Stress","xigua_live_test").\n' \
                             '		check(status.is(200)))\n' \
                             '	setUp(scn.inject(\n'

    script_front['chat'] = 'package computerdatabase.ImChatTest\n' + common_input + \
                        'class ImChatTest extends Simulation {\n' \
                        'var http_conf = http.baseURL("http://i.snssdk.com")\n' \
                        'val didFeeder = csv("did.csv").random\n' \
                        'val sessionFeeder = csv("sessionid.csv").random\n' \
                        'val scn = scenario("ImChatTest").feed(didFeeder).feed(sessionFeeder).\n' \
                        '		exec(http("ImChatTest").\n' \
                        '		post("/videolive/im/chat?room_id='+room_id+'&sdk_version=120&iid=46732565125&device_id=${did}&ac=wifi&channel=huawei&aid=32&app_name=video_article&version_code=702&version_name=7.0.2&device_platform=android&ab_version=544933%2C321290%2C425684%2C549910%2C545153%2C554456%2C437002%2C539907%2C521142%2C544698%2C521140%2C357767%2C551966%2C523413%2C542501%2C548493%2C442144%2C487822%2C374103%2C536394%2C489304%2C539966%2C378354%2C553336%2C552063%2C459657%2C506536%2C545048%2C553899%2C457536%2C543415%2C552304%2C551242%2C344692&ssmix=a&device_type=FRD-AL10&device_brand=honor&language=zh&os_api=24&os_version=7.0&uuid=863127036074315&openudid=be61d330e8faf172&manifest_version_code=302&resolution=1080*1794&dpi=480&update_version_code=70208&_rticket=1539941348352&fp=LlTqL2X5crDrFlHuJ2U1FlceL2cu&rom_version=EmotionUI_5.0_FRD-AL10C00B387").\n' \
                        '		header("Cookie"," sessionid=${sessionid}").\n' \
                        '		header("X-Tt-Stress","xigua_live_test").\n' \
                        '		formParam("content", "46458970920").\n' \
                        '		check(status.is(200)))\n' \
                        '	setUp(scn.inject(\n'

    def write_script(self, key, stage, file_path):
        with open(file_path, 'w') as f:
            key_params = "rampUsersPerSec(1) to(%d) during(60),constantUsersPerSec(%d) during(600)" \
                           % (self.stage_params[key] * stage, self.stage_params[key] * stage)
            content = self.script_front[key] + key_params + self.script_end
            f.write(content)

    def write_scripts(self, stage):
        self.write_script('rooms', stage, os.path.join(file_path, self.file_name['rooms']))
        self.write_script('audience_room_info', stage, os.path.join(file_path, self.file_name['audience_room_info']))
        self.write_script('get_msg', stage, os.path.join(file_path, self.file_name['get_msg']))
        self.write_script('rank_list', stage, os.path.join(file_path, self.file_name['rank_list']))
        self.write_script('chat', stage, os.path.join(file_path, self.file_name['chat']))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('参数错误')
    else:
        stage = sys.argv[1]
        ws = WriteScripts()
        ws.write_scripts(int(stage))


