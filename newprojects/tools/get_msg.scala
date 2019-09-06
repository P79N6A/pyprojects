package computerdatabase.GetMsgTest
import java.util
import java.util.Collections
import java.util.concurrent.ThreadLocalRandom
import io.gatling.core.Predef._
import io.gatling.http.Predef._
import scala.collection.mutable.ArraySeq
import io.gatling.core.structure.PopulationBuilder
import org.asynchttpclient.{Param, RequestBuilderBase}
class GetMsgTest extends Simulation {
var http_conf = http.baseURL("http://i.snssdk.com")
val didFeeder = csv("did.csv").random
val scn = scenario("GetMsgTest").feed(didFeeder).
				exec(http("GetMsgTest").
				get("/videolive/im/get_msg?cursor=0&room_id=6626917226559572749&&sdk_version=120&iid=46732565125&device_id=${did}&ac=wifi&channel=huawei&aid=32&app_name=video_article&version_code=702&version_name=7.0.2&device_platform=android&ab_version=544933%2C321290%2C425684%2C549910%2C545153%2C554456%2C437002%2C539907%2C521142%2C544698%2C521140%2C357767%2C551966%2C523413%2C542501%2C548493%2C442144%2C487822%2C374103%2C536394%2C489304%2C539966%2C378354%2C553336%2C552063%2C459657%2C506536%2C545048%2C553899%2C457536%2C543415%2C552304%2C551242%2C344692&ssmix=a&device_type=FRD-AL10&device_brand=honor&language=zh&os_api=24&os_version=7.0&uuid=863127036074315&openudid=be61d330e8faf172&manifest_version_code=302&resolution=1080*1794&dpi=480&update_version_code=70208&_rticket=1539941348352&fp=LlTqL2X5crDrFlHuJ2U1FlceL2cu&rom_version=EmotionUI_5.0_FRD-AL10C00B387").
				header("X-Tt-Stress","xigua_live_test").
				check(status.is(200)))
	setUp(scn.inject(
rampUsersPerSec(1) to(100000) during(60),constantUsersPerSec(100000) during(600)).protocols(http_conf))
}
