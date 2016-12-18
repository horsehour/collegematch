package edu.rpi;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeoutException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.rabbitmq.client.AMQP;
import com.rabbitmq.client.Channel;
import com.rabbitmq.client.Connection;
import com.rabbitmq.client.ConnectionFactory;
import com.rabbitmq.client.Consumer;
import com.rabbitmq.client.DefaultConsumer;
import com.rabbitmq.client.Envelope;

import edu.rpi.util.DataUtil;
import redis.clients.jedis.Jedis;
import redis.clients.jedis.JedisPool;
import redis.clients.jedis.JedisPoolConfig;

public class MessengeCenter {
	private static final Logger logger = LoggerFactory.getLogger(MessengeCenter.class);
	private final static String QUEUE_NAME = "predictor_queue";

	JedisPool pool = null;
	CollegeMatch cm = null;

	public final int portRedis = 6379;

	public MessengeCenter() {
		cm = new CollegeMatch();
	}

	void train(String[] args, Jedis redis) {
		if (args.length >= 5) {
			double[][] colleges = DataUtil.getRedisData(redis, args[1], true);
			double[][] students = DataUtil.getRedisData(redis, args[2], true);
			List<int[]> preferences = new ArrayList<>();
			DataUtil.getRedisData(redis, args[3], preferences);
			double[][] w = cm.learn(colleges, students, preferences);
			DataUtil.setRedisData(redis, args[4], w, false);
		} else
			logger.info("USAGE: -train <colleges> <students> <preferences> <model>");
	}

	void predict(String[] args, Jedis redis) {
		if (args.length >= 6) {
			double[][] w = DataUtil.getRedisData(redis, args[1], false);
			double[][] colleges = DataUtil.getRedisData(redis, args[2], true);
			double[] student = new double[w.length];
			DataUtil.getRedisData(redis, args[3], true, student);
			int k = Integer.parseInt(args[4]);
			int[] preference = null;
			if (k <= 0)
				preference = cm.predict(student, colleges, w);
			else
				preference = cm.predict(student, colleges, w, k);
			DataUtil.setRedisData(redis, args[5], preference);
		} else
			logger.info("USAGE: -predict <model> <colleges> <student> <k> <preference>");
	}

	void enhance(String[] args, Jedis redis) {
		if (args.length >= 6) {
			double[][] w = DataUtil.getRedisData(redis, args[1], false);
			double[][] colleges = DataUtil.getRedisData(redis, args[2], true);
			double[][] students = DataUtil.getRedisData(redis, args[3], true);
			List<int[]> preferences = new ArrayList<>();
			DataUtil.getRedisData(redis, args[4], preferences);
			w = cm.enhance(w, colleges, students, preferences);
			DataUtil.setRedisData(redis, args[5], w, false);
		} else
			logger.info("USAGE: -enhance <model> <colleges> <students> <preferences> <model_enhanced>");
	}

	class QueryConsumer extends DefaultConsumer {
		private final Logger loggerQuery = LoggerFactory.getLogger(MessengeCenter.class);

		public QueryConsumer(Channel channel) {
			super(channel);
		}

		@Override
		public void handleDelivery(String consumerTag, Envelope envelope, AMQP.BasicProperties properties, byte[] body)
				throws IOException {
			String request = new String(body, "UTF8");
			String[] args = request.split(" ");
			String prefix = args[0];
			try (Jedis redis = pool.getResource()) {
				if (prefix.startsWith("-train"))
					train(args, redis);
				else if (prefix.startsWith("-predict"))
					predict(args, redis);
				else if (prefix.startsWith("-enhance"))
					enhance(args, redis);
			}
			loggerQuery.info("Received " + request);
		}
	}

	/**
	 * Started up RabbitMQ listener
	 */
	public void run(String hostRedis, String hostRabbitMQ) {
		if (hostRedis == null || hostRedis.isEmpty())
			hostRedis = "localhost";
		if (hostRabbitMQ == null || hostRabbitMQ.isEmpty())
			hostRabbitMQ = "localhost";

		cm = new CollegeMatch();
		pool = new JedisPool(new JedisPoolConfig(), hostRedis, portRedis);

		ConnectionFactory factory = new ConnectionFactory();
		factory.setHost(hostRabbitMQ);

		Connection connection;
		try {
			connection = factory.newConnection();
			Channel channel = connection.createChannel();

			boolean durable = false;
			channel.queueDeclare(QUEUE_NAME, durable, false, false, null);

			logger.info("Waiting for messages from " + QUEUE_NAME);
			Consumer consumer = new QueryConsumer(channel);

			boolean autoAck = true;
			// loop that waits for message
			channel.basicConsume(QUEUE_NAME, autoAck, consumer);
		} catch (IOException e) {
			logger.info("RabbitMQ server is down.");
			logger.info(e.getMessage());
		} catch (TimeoutException e) {
			logger.info(e.getMessage());
		}
	}

	public static void main(String[] args) {
		if (args == null) {
			System.err.println("USAGE:\n -train <colleges> <students> <preferences> <model> <redis_host>");
			System.err.println(" -predict <model> <colleges> <student> <k> <preference> <redis_host>");
			System.err.println(" -enhance <model> <colleges> <students> <preferences> <model_enhanced> <redis_host>");
			return;
		}
		int m = args.length;
		String hostRedis = "";
		if (m > 1)
			hostRedis = args[m - 1];

		MessengeCenter mc = new MessengeCenter();
		JedisPool pool = new JedisPool(new JedisPoolConfig(), hostRedis, mc.portRedis);

		String prefix = args[0];
		try (Jedis redis = pool.getResource()) {
			if (prefix.startsWith("-train"))
				mc.train(args, redis);
			else if (prefix.startsWith("-predict"))
				mc.predict(args, redis);
			else if (prefix.startsWith("-enhance"))
				mc.enhance(args, redis);
		}

		if (!pool.isClosed())
			pool.close();
		pool.destroy();
	}
}
