/**
 * @format
 */

import React from 'react';
import ReactTestRenderer from 'react-test-renderer';
import App from '../App';

jest.mock('react-native-safe-area-context', () => {
  const React = require('react');
  const { View } = require('react-native');

  return {
    SafeAreaProvider: ({ children }) =>
      React.createElement(React.Fragment, null, children),
    SafeAreaView: View,
  };
});

const getTab = (renderer: ReactTestRenderer.ReactTestRenderer, label: string) =>
  renderer.root
    .findAllByProps({ accessibilityLabel: label })
    .find((tab) => typeof tab.props.onPress === 'function');

test('renders correctly', async () => {
  await ReactTestRenderer.act(() => {
    ReactTestRenderer.create(<App />);
  });
});

test('keeps other tabs locked until Get started is pressed', async () => {
  let renderer: ReactTestRenderer.ReactTestRenderer;
  await ReactTestRenderer.act(() => {
    renderer = ReactTestRenderer.create(<App />);
  });

  expect(getTab(renderer!, 'Home')?.props.accessibilityState.selected).toBe(true);
  expect(getTab(renderer!, 'Streak')?.props.disabled).toBe(true);
  expect(getTab(renderer!, 'FAQ')?.props.disabled).toBe(true);
  expect(getTab(renderer!, 'Home')?.props.accessibilityState.selected).toBe(true);

  await ReactTestRenderer.act(() => {
    getTab(renderer!, 'Get started')?.props.onPress();
  });

  expect(getTab(renderer!, 'Streak')?.props.disabled).toBe(false);
  expect(getTab(renderer!, 'FAQ')?.props.disabled).toBe(false);

  await ReactTestRenderer.act(() => {
    getTab(renderer!, 'FAQ')?.props.onPress();
  });

  expect(getTab(renderer!, 'FAQ')?.props.accessibilityState.selected).toBe(true);
  expect(getTab(renderer!, 'Home')?.props.accessibilityState.selected).toBe(false);
});
