package edu.rpi;

import redis.clients.jedis.Jedis;
import redis.clients.jedis.JedisPool;
import redis.clients.jedis.JedisPoolConfig;

/**
 *
 * @author Chunheng Jiang
 * @version 1.0
 * @since 11:20:16 AM, Dec 1, 2016
 */

public class Redis {
	public String host = "localhost";
	public int port = 6379;

	public Redis() {
	}

	/**
	 * Build connection with redis db
	 * 
	 * @param host
	 * @return JedisPool instance
	 */
	public JedisPool connectRedis(String host, int port) {
		/* redis-cli connects to the server at port 6379 */
		// host: 104.236.86.53
		JedisPool pool = new JedisPool(new JedisPoolConfig(), host, port);
		return pool;
	}

	public JedisPool connectRedis(String host) {
		return connectRedis(host, port);
	}

	public JedisPool connectRedis() {
		return connectRedis(host, port);
	}

	/**
	 * Read value from redis with a key
	 * 
	 * @param host
	 * @param key
	 * @param close
	 * @return value
	 */
	public String readRedis(String host, String key, boolean close) {
		JedisPool pool = connectRedis(host);
		String value = "";
		try (Jedis redis = pool.getResource()) {
			value = redis.get(key);
		}

		if (close) {
			if (!pool.isClosed())
				pool.close();
			pool.destroy();
		}
		return value;
	}

	public void writeRedis(String key, String val, Jedis jedis) {
		jedis.set(key, val);
	}

	public static void main(String[] args) {
		Redis redis = new Redis();
		String host = "104.236.86.53";
		String output = redis.readRedis(host, "learning:college_training.csv", true);

		System.out.println(output);
	}
}
