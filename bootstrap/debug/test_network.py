# test_network.py
import socket
import requests
from config.config import config


def test_connection():
    host = config.ES_HOST
    port = config.ES_PORT

    print("=" * 60)
    print(f"测试连接: {host}:{port}")
    print("=" * 60)

    # 1. 测试基本的网络连通性
    print("1. 测试网络连通性...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()

        if result == 0:
            print(f"   ✅ 可以连接到 {host}:{port}")
        else:
            print(f"   ❌ 无法连接到 {host}:{port} (错误代码: {result})")
            print("   可能原因:")
            print("   - 服务器未运行")
            print("   - 防火墙阻止")
            print("   - 服务器IP错误")
            return False
    except Exception as e:
        print(f"   ❌ 连接测试异常: {e}")
        return False

    # 2. 测试HTTP连接
    print("\n2. 测试HTTP连接...")
    try:
        response = requests.get(f"http://{host}:{port}", timeout=5, verify=False)
        print(f"   HTTP状态码: {response.status_code}")

        if response.status_code == 401:
            print("   ⚠️ 需要认证")
            print(f"   当前配置的用户名: {config.ES_USERNAME}")
            print(f"   当前配置的密码: {'*' * len(config.ES_PASSWORD) if config.ES_PASSWORD else '未设置'}")
        elif response.status_code == 200:
            data = response.json()
            print(f"   ✅ 连接成功")
            print(f"   集群: {data.get('cluster_name', '未知')}")
            print(f"   版本: {data.get('version', {}).get('number', '未知')}")
        else:
            print(f"   ❌ 非预期状态码: {response.status_code}")

    except requests.exceptions.Timeout:
        print("   ⏱️  HTTP连接超时")
    except requests.exceptions.ConnectionError:
        print("   ❌ HTTP连接被拒绝")
    except Exception as e:
        print(f"   ❌ HTTP连接错误: {e}")

    return True


if __name__ == "__main__":
    test_connection()