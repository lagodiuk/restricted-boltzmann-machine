package com.lagodiuk.rbm;

import java.util.Arrays;

public class RBM {

	private double[] v; // visible

	private double[] h; // hidden

	private double[][] w; // weights

	private double[] vBiases;

	private double[] hBiases;

	public RBM(int vDim, int hDim) {
		this.v = new double[vDim];
		this.h = new double[hDim];
		this.w = new double[vDim][hDim];
		for (int i = 0; i < vDim; i++) {
			for (int j = 0; j < hDim; j++) {
				this.w[i][j] = this.rand();
			}
		}
		this.vBiases = new double[vDim];
		this.hBiases = new double[hDim];
	}

	public void setVisible(double[] input) {
		System.arraycopy(input, 0, this.v, 0, this.v.length);
	}

	public double[] getVisible() {
		return this.v;
	}

	public void think(int times) {
		for (int i = 0; i < times; i++) {
			this.calculateHidden();
			this.calculateVisible();
		}
	}

	private void calculateHidden() {
		this.activate(this.v, this.h, this.hBiases, this.w, true);
	}

	private void calculateVisible() {
		this.activate(this.h, this.v, this.vBiases, this.w, false);
	}

	private void activate(double[] src, double[] dest, double[] destBiases,
			double[][] visibleHiddenWeights, boolean fromVisibleToHidden) {
		for (int i = 0; i < dest.length; i++) {
			double signal = 0;
			for (int j = 0; j < src.length; j++) {
				double w = fromVisibleToHidden ? visibleHiddenWeights[j][i]
						: visibleHiddenWeights[i][j];
				signal += src[j] * w;
			}
			signal += destBiases[i];
			double probability = this.sigma(signal);
			if (this.rand() < probability) {
				dest[i] = 1;
			} else {
				dest[i] = 0;
			}
			// dest[i] = probability;
		}
	}

	public void remember(double[] input, double learningRate) {
		this.setVisible(input);

		this.calculateHidden();

		double[][] positiveGradient = this.outerProductVH();

		this.think(10);

		double[][] negativeGradient = this.outerProductVH();

		double[] hSnapshot = new double[this.h.length];
		System.arraycopy(this.h, 0, hSnapshot, 0, this.h.length);

		for (int i = 0; i < this.v.length; i++) {
			for (int j = 0; j < this.h.length; j++) {
				this.w[i][j] += (positiveGradient[i][j] - negativeGradient[i][j])
						* learningRate;
			}
		}

		for (int i = 0; i < this.v.length; i++) {
			this.vBiases[i] += (input[i] - this.v[i]) * learningRate;
		}

		for (int i = 0; i < this.h.length; i++) {
			this.hBiases[i] += (hSnapshot[i] - this.h[i]) * learningRate;
		}
	}

	private double[][] outerProductVH() {
		double[][] res = new double[this.v.length][this.h.length];
		for (int i = 0; i < this.v.length; i++) {
			for (int j = 0; j < this.h.length; j++) {
				res[i][j] = this.v[i] * this.h[j];
			}
		}
		return res;
	}

	private double sigma(double x) {
		return 1.0 / (1.0 + Math.exp(-x));
	}

	private double rand() {
		return Math.random();
	}

	public static void main(String[] args) {
		double[][] trainingSet = { { 0, 1, 1, 1, 1 }, { 1, 1, 0, 0, 0 },
				{ 1, 0, 0, 0, 1 } };

		RBM rbm = new RBM(trainingSet[0].length, 3);

		for (int k = 0; k < 10000; k++) {
			for (int i = 0; i < trainingSet.length; i++) {
				rbm.remember(trainingSet[i], 0.1);
			}
		}

		for (int k = 0; k < 10; k++) {
			rbm.setVisible(new double[] { 1, 1, 0, 0, 0 });
			rbm.think(10);
			System.out.println(Arrays.toString(rbm.getVisible()));
		}
	}
}
