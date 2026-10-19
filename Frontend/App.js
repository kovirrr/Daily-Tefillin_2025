import React, { useState } from 'react';
import { Pressable, Text, View } from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import FaqPage from './components/faq_page/faq_page';
import StreakPage from './components/streak_page/streak_page';
import styles from './styles/appStyles';

const tabs = [
  { key: 'streak', label: 'Streak' },
  { key: 'home', label: 'Home' },
  { key: 'faq', label: 'FAQ' },
];

function App() {
  const [activeTab, setActiveTab] = useState('home');
  const [navigationUnlocked, setNavigationUnlocked] = useState(false);

  let screen;
  if (activeTab === 'faq') {
    screen = <FaqPage />;
  } else if (activeTab === 'streak') {
    screen = <StreakPage />;
  } else {
    screen = (
      <View style={styles.screen}>
        <View style={styles.homeContent}>
          <Text style={styles.screenTitle}>Daily Tefillin</Text>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel={
              navigationUnlocked ? 'Navigation unlocked' : 'Get started'
            }
            disabled={navigationUnlocked}
            onPress={() => setNavigationUnlocked(true)}
            style={[
              styles.primaryButton,
              navigationUnlocked && styles.primaryButtonDisabled,
            ]}
          >
            <Text style={styles.primaryButtonLabel}>
              {navigationUnlocked ? 'Ready to go' : 'Get started'}
            </Text>
          </Pressable>
        </View>
      </View>
    );
  }

  return (
    <SafeAreaProvider>
      <SafeAreaView style={styles.app} edges={['top', 'bottom']}>
        <View style={styles.screenContainer}>{screen}</View>
        <View style={styles.bottomNav} accessibilityRole="tablist">
          {tabs.map((tab) => {
            const isActive = activeTab === tab.key;
            const isLocked = tab.key !== 'home' && !navigationUnlocked;

            return (
              <Pressable
                key={tab.key}
                accessibilityLabel={tab.label}
                accessibilityRole="tab"
                accessibilityState={{ selected: isActive, disabled: isLocked }}
                disabled={isLocked}
                onPress={() => setActiveTab(tab.key)}
                style={[
                  styles.navButton,
                  isActive && styles.navButtonActive,
                  isLocked && styles.navButtonLocked,
                ]}
              >
                <Text
                  style={[
                    styles.navLabel,
                    isActive && styles.navLabelActive,
                    isLocked && styles.navLabelLocked,
                  ]}
                >
                  {tab.label}
                </Text>
              </Pressable>
            );
          })}
        </View>
      </SafeAreaView>
    </SafeAreaProvider>
  );
}

export default App;
